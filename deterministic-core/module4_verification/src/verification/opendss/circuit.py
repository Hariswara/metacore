"""OpenDSS Circuit Twin Wrapper.

ZERO ML DEPENDENCIES. Sole interface to OpenDSSDirect.py.
"""

from pathlib import Path

import opendssdirect as dss

DEFAULT_MODEL_PATH = Path(__file__).parent / "models" / "delft_3island.dss"
DEFAULT_TOPOLOGY_ID = "delft-3island-v0"

_MISSING = "QUALITY_MISSING"


class CircuitTwin:
    """Manages an OpenDSS microgrid simulation instance and state extraction."""

    def __init__(self, dss_file_path: str | None = None, topology_id: str | None = None) -> None:
        self.dss_file_path = Path(dss_file_path) if dss_file_path else DEFAULT_MODEL_PATH
        # Which GridTopology this twin instantiates. A GridStateSnapshot for any other topology
        # is refused rather than half-applied (see apply_snapshot).
        self.topology_id = topology_id or (
            DEFAULT_TOPOLOGY_ID if self.dss_file_path == DEFAULT_MODEL_PATH
            else self.dss_file_path.stem
        )
        self._initial_load_cache: dict[str, tuple[float, float]] = {}
        self._initial_gen_cache: dict[str, tuple[float, float]] = {}
        self.compile_base_circuit()

    def compile_base_circuit(self) -> bool:
        """Compiles or resets the circuit to its baseline definition."""
        dss.Command(f'compile "{self.dss_file_path}"')
        self._cache_base_equipment()
        return True

    def _cache_base_equipment(self) -> None:
        """Caches nominal generation and load ratings for relative adjustments."""
        self._initial_load_cache.clear()
        for load_name in dss.Loads.AllNames():
            dss.Loads.Name(load_name)
            p = float(dss.Loads.kW())
            q = float(dss.Loads.kvar())
            self._initial_load_cache[load_name.lower()] = (p, q)

        self._initial_gen_cache.clear()
        for gen_name in dss.Generators.AllNames():
            dss.Generators.Name(gen_name)
            p = float(dss.Generators.kW())
            q = float(dss.Generators.kvar())
            self._initial_gen_cache[gen_name.lower()] = (p, q)

    def reset_to_base(self) -> None:
        """Restores circuit to nominal baseline state."""
        self.compile_base_circuit()

    def apply_snapshot(self, snapshot: dict) -> list[str]:
        """Loads an M1 GridStateSnapshot (module1.proto) onto the twin.

        Sets feeder-bus demand from every node state whose p_kw is not QUALITY_MISSING, and
        disables the assets the snapshot lists as out of service. Fields the snapshot marks
        missing are left at the model's base value -- never zeroed. The applied state becomes
        the new reference for load shedding, so a shed fraction is a fraction of *this hour's*
        demand. Returns a list of problems; any problem means the snapshot was not applied
        consistently and the caller must reject.
        """
        problems: list[str] = []
        if snapshot.get("topology_id") != self.topology_id:
            return [f"snapshot topology {snapshot.get('topology_id')!r} != twin "
                    f"{self.topology_id!r}"]

        loads_by_bus: dict[str, list[str]] = {}
        for load_name in dss.Loads.AllNames():
            dss.Circuit.SetActiveElement(f"Load.{load_name}")
            bus = dss.CktElement.BusNames()[0].split(".")[0].lower()
            loads_by_bus.setdefault(bus, []).append(load_name)

        for ns in snapshot.get("node_states", []):
            quality = ns.get("quality", {})
            if ns.get("p_kw") is None or quality.get("p_kw") == _MISSING:
                continue
            names = loads_by_bus.get(str(ns["node_id"]).lower())
            if not names:
                problems.append(f"no load at bus {ns['node_id']!r}")
                continue
            p_total = float(ns["p_kw"])
            q_total = float(ns["q_kvar"]) if ns.get("q_kvar") is not None else None
            base = [self._initial_load_cache.get(n.lower(), (0.0, 0.0)) for n in names]
            base_p = sum(p for p, _ in base)
            for name, (p0, _q0) in zip(names, base, strict=True):
                # Split across the bus's loads in proportion to their base demand.
                w = (p0 / base_p) if base_p else 1.0 / len(names)
                dss.Loads.Name(name)
                dss.Loads.kW(p_total * w)
                if q_total is not None:
                    dss.Loads.kvar(q_total * w)

        generators = {g.lower() for g in dss.Generators.AllNames()}
        for element in snapshot.get("assets_out_of_service", []):
            if element.lower() not in generators:
                problems.append(f"no generator {element!r} to take out of service")
                continue
            dss.Circuit.SetActiveElement(f"Generator.{element}")
            dss.CktElement.Enabled(False)

        self._cache_base_equipment()
        return problems

    def get_all_buses(self) -> list[str]:
        """Returns all bus names in the active circuit."""
        return [b.upper() for b in dss.Circuit.AllBusNames()]

    def get_all_lines(self) -> list[str]:
        """Returns all line names in the active circuit."""
        return [line.upper() for line in dss.Lines.AllNames()]

    def get_all_generators(self) -> list[str]:
        """Returns all generator names in the active circuit."""
        return [gen.upper() for gen in dss.Generators.AllNames()]

    def get_all_loads(self) -> list[str]:
        """Returns all load names in the active circuit."""
        return [load.upper() for load in dss.Loads.AllNames()]

    def get_bus_voltages_pu(self) -> dict[str, float]:
        """Returns minimum per-unit voltage magnitude for each bus across all active phases."""
        bus_voltages: dict[str, float] = {}
        for bus_name in dss.Circuit.AllBusNames():
            dss.Circuit.SetActiveBus(bus_name)
            pu_mags = dss.Bus.puVmagAngle()
            if pu_mags and len(pu_mags) >= 2:
                # puVmagAngle returns [mag1, ang1, mag2, ang2, ...]
                mags = [float(pu_mags[i]) for i in range(0, len(pu_mags), 2)]
                if mags:
                    bus_voltages[bus_name.upper()] = min(mags)
        return bus_voltages

    def get_line_loadings(self) -> dict[str, dict[str, float]]:
        """Returns line current magnitudes and rating margins."""
        line_data: dict[str, dict[str, float]] = {}
        for line_name in dss.Lines.AllNames():
            dss.Lines.Name(line_name)
            dss.Circuit.SetActiveElement(f"Line.{line_name}")
            norm_amps = float(dss.Lines.NormAmps())
            is_enabled = bool(dss.CktElement.Enabled())
            currents = dss.CktElement.CurrentsMagAng()
            max_current = 0.0
            if currents and len(currents) >= 2 and is_enabled:
                mags = [float(currents[i]) for i in range(0, min(6, len(currents)), 2)]
                if mags:
                    max_current = max(mags)

            margin_fraction = (max_current - norm_amps) / norm_amps if norm_amps > 0 else 0.0
            line_data[line_name.upper()] = {
                "max_amps": max_current,
                "norm_amps": norm_amps,
                "margin_fraction": margin_fraction,
                "enabled": 1.0 if is_enabled else 0.0,
            }
        return line_data

    def set_line_state(self, edge_id: str, closed: bool) -> bool:
        """Toggles a line or tie-breaker switch."""
        for line_name in dss.Lines.AllNames():
            if line_name.lower() == edge_id.lower():
                dss.Circuit.SetActiveElement(f"Line.{line_name}")
                dss.CktElement.Enabled(closed)
                return True
        return False

    def set_load_shed(self, node_id: str, shed_fraction: float) -> bool:
        """Scales active and reactive load at a target node bus."""
        matched = False
        shed_fraction = max(0.0, min(1.0, shed_fraction))
        for load_name in dss.Loads.AllNames():
            dss.Loads.Name(load_name)
            dss.Circuit.SetActiveElement(f"Load.{load_name}")
            bus_name = dss.CktElement.BusNames()[0].split(".")[0]
            if bus_name.lower() == node_id.lower() or node_id.lower() in load_name.lower():
                base_fallback = (float(dss.Loads.kW()), float(dss.Loads.kvar()))
                base_p, base_q = self._initial_load_cache.get(load_name.lower(), base_fallback)
                new_p = base_p * (1.0 - shed_fraction)
                new_q = base_q * (1.0 - shed_fraction)
                dss.Loads.kW(new_p)
                dss.Loads.kvar(new_q)
                matched = True
        return matched

    def set_generator_dispatch(self, node_id: str, p_kw: float, q_kvar: float = 0.0) -> bool:
        """Updates active (kW) and reactive (kvar) generation at a target node bus."""
        matched = False
        for gen_name in dss.Generators.AllNames():
            dss.Generators.Name(gen_name)
            dss.Circuit.SetActiveElement(f"Generator.{gen_name}")
            bus_name = dss.CktElement.BusNames()[0].split(".")[0]
            if bus_name.lower() == node_id.lower() or node_id.lower() in gen_name.lower():
                dss.Generators.kW(p_kw)
                dss.Generators.kvar(q_kvar)
                matched = True
        return matched
