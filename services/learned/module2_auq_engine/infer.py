"""The runtime consumer: what M3 actually calls, one state at a time.

    from infer import OnnxAUQ
    auq = OnnxAUQ.load()
    auq.calibrate(id_features)                  # once, against in-distribution states
    auq.score(state.node_features[0], state.quality.observed_fraction)

Everything the frozen graph does not do lives here, and that split is the point:

  in the ONNX graph   the network -> evidence
  here                normalisation, u = K/S, the observed_fraction discount, the
                      two-condition trigger, and the M2->M3 message

`observed_fraction` arrives with each state from M1's QualityMask, and the thresholds are
calibrated per deployment, so neither belongs in an artifact you would otherwise have to
re-export to retune.

Requires the `onnx` extra. Import is lazy so the module is safe to import without it.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from contract import build_output
from trigger import CompetenceDropTrigger

HERE = Path(__file__).parent
MODEL_PATH = HERE / "edl.onnx"
CARD_PATH = HERE / "edl.onnx.json"


class OnnxAUQ:
    """Batch-1 scoring against the exported evidence graph."""

    def __init__(self, session, card, of_floor, hysteresis, value_threshold=None):
        self._session = session
        self.card = card
        self.mu = np.asarray(card["normalisation"]["mean"], dtype=np.float32)
        self.sd = np.asarray(card["normalisation"]["std"], dtype=np.float32)
        self.k = int(card["architecture"]["classes"])
        self.feature_names = list(card["architecture"]["feature_names"])
        self.of_floor = of_floor
        self.hysteresis = hysteresis
        self.value_threshold = value_threshold
        self._trigger = None

    @classmethod
    def load(cls, model_path=MODEL_PATH, card_path=CARD_PATH, value_threshold=None):
        import onnxruntime as ort

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} not found; run `python export_onnx.py`")
        with open(HERE / "config.yaml") as fh:
            cfg = yaml.safe_load(fh)
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        card = json.loads(Path(card_path).read_text())
        auq = cls(session, card, cfg["trigger"]["observed_fraction_floor"],
                  cfg["trigger"]["hysteresis"], value_threshold)
        auq._reset_trigger()
        return auq

    def _reset_trigger(self):
        self._trigger = CompetenceDropTrigger(
            self.value_threshold if self.value_threshold is not None else float("inf"),
            self.of_floor, self.hysteresis)

    # ------------------------------------------------------------------ core

    def evidence(self, features):
        """One state's raw features -> evidence. Normalisation applied here, since the
        graph expects standardised input and the statistics live in the model card."""
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features {self.feature_names}, "
                f"got {x.shape[1]}")
        return self._session.run(None, {"state": (x - self.mu)/self.sd})[0]

    def uncertainty(self, evidence, observed_fraction=1.0):
        """(value_u, quality_adjusted_u, class_probabilities) for one state.

        Mirrors edl.uncertainty / edl.uncertainty_quality in NumPy so the serving path
        carries no torch dependency.
        """
        total = float(evidence.sum())
        alpha = evidence + 1.0
        probs = (alpha/alpha.sum())[0]
        value_u = self.k/(total + self.k)
        adjusted_u = self.k/(total*float(observed_fraction) + self.k)
        return value_u, adjusted_u, probs

    def calibrate(self, id_features, value_false_alarm_rate=0.05):
        """Fit the value threshold on plain value-u over in-distribution states.

        Quality-independent by construction -- see CompetenceDropTrigger.calibrate.
        """
        value_u = np.array([self.uncertainty(self.evidence(f))[0] for f in id_features])
        trig = CompetenceDropTrigger.calibrate(
            value_u, value_false_alarm_rate, self.of_floor, self.hysteresis)
        self.value_threshold = trig.vthr
        self._reset_trigger()
        return self.value_threshold

    def score(self, features, observed_fraction, aleatoric_proxy=0.0):
        """One state in, one M2Output out. Stateful: the trigger debounces across calls."""
        if self.value_threshold is None:
            raise RuntimeError("call calibrate() or pass value_threshold= before scoring")
        value_u, adjusted_u, probs = self.uncertainty(self.evidence(features),
                                                      observed_fraction)
        fired, reason = self._trigger.update(value_u, observed_fraction)
        return build_output(adjusted_u, probs, aleatoric_proxy, fired,
                            observed_fraction, reason)

    def reset(self):
        """Clear the trigger's hysteresis state between episodes."""
        self._reset_trigger()
