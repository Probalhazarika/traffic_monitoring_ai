# ─────────────────────────────────────────────────
#  traffic/signal_controller.py
#  Dynamic green-light timing based on traffic density.
# ─────────────────────────────────────────────────

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (GREEN_LOW, GREEN_MEDIUM, GREEN_HIGH,
                    MIN_GREEN, MAX_GREEN)


class SignalController:
    """
    Assigns green-light durations to each lane.

    Two strategies are available:
      1. density_based  – simple lookup from density label
      2. proportional   – weighted by vehicle count relative to total

    The dashboard uses `compute_timings()` which returns both plus
    the current phase (which lane is currently GREEN).
    """

    # ── Strategy 1: Density lookup ──────────────

    @classmethod
    def density_based_timing(cls, lane_stats: dict) -> dict:
        """
        Assigns green time directly based on vehicle count.
        - count > 20: 45s
        - count > 15: 35s
        - count > 9 : 25s
        - default   : 15s
        
        Parameters
        ----------
        lane_stats : {lane_name: {"count": int, "density": str}}

        Returns
        -------
        {lane_name: green_seconds}
        """
        timings = {}
        for lane, info in lane_stats.items():
            count = info["count"]
            if count > 20:
                t = 45
            elif count > 15:
                t = 35
            elif count > 9:
                t = 25
            else:
                t = 15
            timings[lane] = t
        return timings


    # ── Strategy 2: Proportional timing ─────────

    @staticmethod
    def proportional_timing(lane_stats: dict,
                            total_cycle: int = 120) -> dict:
        """
        Distributes `total_cycle` seconds proportionally across lanes.
        Lanes with zero vehicles get the minimum green time.

        Parameters
        ----------
        lane_stats   : {lane_name: {"count": int, ...}}
        total_cycle  : total seconds for one full signal cycle

        Returns
        -------
        {lane_name: green_seconds}
        """
        counts = {lane: max(info["count"], 1)   # avoid div-by-zero
                  for lane, info in lane_stats.items()}
        total  = sum(counts.values())

        timings = {}
        for lane, cnt in counts.items():
            raw = (cnt / total) * total_cycle
            timings[lane] = int(max(MIN_GREEN, min(MAX_GREEN, raw)))

        return timings

    # ── Main API ─────────────────────────────────

    def compute_timings(self, lane_stats: dict) -> dict:
        """
        Combines both strategies and returns a rich signal schedule.

        Returns
        -------
        {
          lane_name: {
            "count"         : int,
            "density"       : str,
            "green_density" : int,   # density-based seconds
            "green_prop"    : int,   # proportional seconds
            "green_time"    : int,   # final recommended (density-based)
            "signal"        : str,   # "GREEN" or "RED"
          }
        }
        plus a separate "phase_order" key listing lanes in desc priority.
        """
        density_t = self.density_based_timing(lane_stats)
        prop_t    = self.proportional_timing(lane_stats)

        # Sort lanes by vehicle count descending → determines which lane
        # gets green light first in the physical cycle
        sorted_lanes = sorted(
            lane_stats.keys(),
            key=lambda l: lane_stats[l]["count"],
            reverse=True
        )

        result = {}
        for i, lane in enumerate(lane_stats.keys()):
            result[lane] = {
                "count":         lane_stats[lane]["count"],
                "density":       lane_stats[lane]["density"],
                "green_density": density_t[lane],
                "green_prop":    prop_t[lane],
                "green_time":    density_t[lane],   # default strategy
                "signal":        "GREEN" if lane == sorted_lanes[0] else "RED",
            }

        result["__phase_order__"] = sorted_lanes
        return result
