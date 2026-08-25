"""M2 -> M3 output contract (see master plan register). Saabir's gate (M3 FR1)
builds against mock_stream() until this module's real output is live."""
import json
import time
from dataclasses import asdict, dataclass

SCHEMA_VERSION = "m2-out/0.2"


@dataclass
class M2Output:
    timestamp: float
    epistemic_uncertainty: float      # quality-aware u in [0,1]  (the gating signal)
    aleatoric_proxy: float            # predictive entropy
    competence_drop: bool             # trigger for M3 to escalate to System 2
    state_class: int                  # argmax safety class (0 safe..K-1 critical)
    class_probabilities: list         # Dirichlet mean p over the K classes
    observed_fraction: float          # share of M1's features that were QUALITY_OBSERVED
    schema_version: str = SCHEMA_VERSION

    def to_json(self):
        return json.dumps(asdict(self))


def build_output(u, p, ent, competence_drop, observed_fraction):
    """`u` is the quality-aware uncertainty; `observed_fraction` is passed through from
    M1's QualityMask so M3 can tell a distribution shift from a sensing failure."""
    return M2Output(timestamp=time.time(),
                    epistemic_uncertainty=float(u),
                    aleatoric_proxy=float(ent),
                    competence_drop=bool(competence_drop),
                    state_class=int(p.argmax()),
                    class_probabilities=[float(v) for v in p],
                    observed_fraction=float(observed_fraction))
