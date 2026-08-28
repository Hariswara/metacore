"""Competence-drop trigger: two independent conditions, OR'd, with hysteresis so it
does not chatter around either boundary.

Folding both axes into one scalar threshold under-fires on sensing loss: the
threshold has to sit above the in-distribution u at M1's *nominal* quality, and a
blackout only pushes u part of the way there. Testing the axes separately fires
reliably on both, and says which one fired -- which M3 wants anyway, because a
distribution shift and a sensing failure deserve different System 2 policies.
"""
import numpy as np

REASON_NONE = "none"
REASON_VALUE = "value"
REASON_SENSING = "sensing"
REASON_BOTH = "both"


class CompetenceDropTrigger:
    """Fires on distribution shift (value-u over threshold) OR sensing loss
    (observed_fraction below floor), and reports which axis fired."""

    def __init__(self, value_threshold, of_floor=0.4, hysteresis=2):
        self.vthr = value_threshold
        self.of_floor = of_floor
        self.h = hysteresis
        self._up = 0
        self._down = 0
        self.state = False
        self.reason = REASON_NONE

    @classmethod
    def calibrate(cls, id_value_u, value_false_alarm_rate=0.05, of_floor=0.4, hysteresis=2):
        """Calibrate on PLAIN value-u, not the quality-adjusted u.

        The value threshold must mean "unusual state", independent of how much of it
        was observed; calibrating it on the combined u would make it drift with M1's
        data quality and reintroduce the under-firing this class exists to fix.
        """
        vthr = float(np.quantile(id_value_u, 1.0 - value_false_alarm_rate))
        return cls(vthr, of_floor, hysteresis)

    def update(self, value_u, observed_fraction):
        v = bool(value_u > self.vthr)
        q = bool(observed_fraction < self.of_floor)
        current = (REASON_BOTH if (v and q)
                   else REASON_VALUE if v
                   else REASON_SENSING if q
                   else REASON_NONE)

        if v or q:
            self._up += 1
            self._down = 0
            if self._up >= self.h:
                self.state = True
        else:
            self._down += 1
            self._up = 0
            if self._down >= self.h:
                self.state = False

        # The reason has to agree with the state M3 is handed. While the trigger is up
        # it reports what is firing, latching through the debounce steps where no
        # condition holds yet the state has not cleared; otherwise it reports none.
        # Without the latch this returns (True, "none") on the way down.
        if self.state:
            self.reason = current if current != REASON_NONE else self.reason
        else:
            self.reason = REASON_NONE
        return self.state, self.reason
