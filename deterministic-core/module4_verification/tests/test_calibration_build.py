"""The offline calibration: persistence-miss table (C5) and the frozen artifact (C7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from verification.calibrate.misses import persistence_misses, read_hourly_load

REPO = Path(__file__).resolve().parents[3]
HOURLY = REPO / "data" / "processed" / "island_load_hourly.csv"

# 10 hours, two islands. Delft doubles over the day; Analaitivu dips then recovers.
HAND = {
    "Delft-Neduntivu": [100, 110, 120, 150, 200, 180, 160, 140, 120, 100],
    "Analaitivu": [50, 40, 45, 50, 55, 60, 50, 40, 30, 50],
}


def write_hand_csv(path: Path) -> Path:
    lines = ["island,timestamp_lst,load_kw,quality"]
    for island, values in HAND.items():
        for h, v in enumerate(values):
            lines.append(f"{island},2025-01-01T{h:02d}:00,{v},QUALITY_INTERPOLATED")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_hand_made_csv_gives_the_expected_misses(tmp_path: Path) -> None:
    load = read_hourly_load(write_hand_csv(tmp_path / "hand.csv"))
    assert load.islands == ("Analaitivu", "Delft-Neduntivu")
    table = persistence_misses(load, "2025-01-01T00:00", "2025-01-01T09:00", horizon=3)
    # 10 hours, a 3-hour window: start hours 00..06
    assert len(table) == 7
    assert table.timestamps[0] == "2025-01-01T00:00" and table.timestamps[-1] == "2025-01-01T06:00"
    delft, ana = 1, 0
    # t=00: Delft 100 -> 110, 120, 150: worst up +50 %, never below
    assert table.up[0, delft] == pytest.approx(0.50)
    assert table.down[0, delft] == pytest.approx(0.10)
    # t=00: Analaitivu 50 -> 40, 45, 50: worst down -20 %, worst up 0 %
    assert table.down[0, ana] == pytest.approx(-0.20)
    assert table.up[0, ana] == pytest.approx(0.0)
    # t=04: Delft 200 -> 180, 160, 140: falls all window
    assert table.up[4, delft] == pytest.approx(-0.10)
    assert table.down[4, delft] == pytest.approx(-0.30)


def test_no_row_uses_data_after_its_own_window(tmp_path: Path) -> None:
    load = read_hourly_load(write_hand_csv(tmp_path / "hand.csv"))
    table = persistence_misses(load, "2025-01-01T00:00", "2025-01-01T05:00", horizon=3)
    # The split ends at 05:00, so the last start hour whose window fits is 02:00
    assert table.timestamps[-1] == "2025-01-01T02:00"
    # Corrupt everything after the split: the table must not change
    later = load.kw.copy()
    later[6:] = 1e6
    from verification.calibrate.misses import HourlyLoad
    poisoned = HourlyLoad(load.timestamps, load.islands, later)
    again = persistence_misses(poisoned, "2025-01-01T00:00", "2025-01-01T05:00", horizon=3)
    assert np.array_equal(again.up, table.up) and np.array_equal(again.down, table.down)


def test_a_gap_in_one_island_is_refused(tmp_path: Path) -> None:
    path = write_hand_csv(tmp_path / "hand.csv")
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError):
        read_hourly_load(path)


def test_real_calibration_split_row_count() -> None:
    load = read_hourly_load(HOURLY)
    assert load.islands == ("Analaitivu", "Delft-Neduntivu", "Eluvaitivu", "Nainativu")
    table = persistence_misses(load, "2024-01-01T00:00", "2025-06-30T23:00")
    hours = (366 + 181) * 24  # 2024 is a leap year; Jan-Jun 2025 has 181 days
    assert len(table) == hours - 6
    assert table.up.shape == table.down.shape == (hours - 6, 4)
    assert np.all(table.up >= table.down)
