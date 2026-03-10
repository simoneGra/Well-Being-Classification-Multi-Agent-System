"""
MobilityAnalysisAgent: Extracts mobility patterns from GPS location data.
No LLM calls - pure geospatial computation in Python.

Computes:
- Radius of gyration (spread of locations)
- Location variance (lat/lng std dev)
- Number of unique location clusters
- Home displacement (distance from residential address)
"""
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict

from agents.data_loader import DataBundle, LocationRecord, UserProfile


@dataclass
class MobilityFeatures:
    citizen_id: str
    n_locations: int = 0
    radius_of_gyration_km: float = 0.0
    lat_std: float = 0.0
    lng_std: float = 0.0
    max_distance_km: float = 0.0
    mean_distance_from_home_km: float = 0.0
    mobility_summary: str = ""


def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Distance in km between two GPS points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class MobilityAnalysisAgent:
    """Analyzes GPS location data to extract mobility features per citizen."""

    def run(self, bundle: DataBundle) -> Dict[str, MobilityFeatures]:
        citizen_locs: Dict[str, List[LocationRecord]] = defaultdict(list)
        for loc in bundle.locations:
            citizen_locs[loc.user_id].append(loc)

        features = {}
        for cid in bundle.citizen_ids:
            locs = citizen_locs.get(cid, [])
            user = bundle.users.get(cid)
            features[cid] = self._compute_mobility(cid, locs, user)

        return features

    def _compute_mobility(
        self,
        cid: str,
        locs: List[LocationRecord],
        user: UserProfile | None,
    ) -> MobilityFeatures:
        mf = MobilityFeatures(citizen_id=cid, n_locations=len(locs))

        if len(locs) < 2:
            mf.mobility_summary = "insufficient location data"
            return mf

        lats = [l.lat for l in locs]
        lngs = [l.lng for l in locs]

        center_lat = statistics.mean(lats)
        center_lng = statistics.mean(lngs)

        mf.lat_std = statistics.stdev(lats)
        mf.lng_std = statistics.stdev(lngs)

        dists_from_center = [_haversine(center_lat, center_lng, lat, lng) for lat, lng in zip(lats, lngs)]
        mf.radius_of_gyration_km = math.sqrt(statistics.mean([d ** 2 for d in dists_from_center]))
        mf.max_distance_km = max(dists_from_center)

        # Distance from home address
        if user and user.lat and user.lng:
            home_dists = [_haversine(user.lat, user.lng, lat, lng) for lat, lng in zip(lats, lngs)]
            mf.mean_distance_from_home_km = statistics.mean(home_dists)

        # Human-readable mobility level
        rog = mf.radius_of_gyration_km
        if rog < 200:
            level = "very_low"
        elif rog < 1000:
            level = "low"
        elif rog < 2000:
            level = "moderate"
        elif rog < 3000:
            level = "high"
        else:
            level = "very_high"

        mf.mobility_summary = (
            f"level={level} rog={rog:.0f}km "
            f"lat_std={mf.lat_std:.3f} lng_std={mf.lng_std:.3f}"
        )

        return mf

    def summarize(self, features: Dict[str, MobilityFeatures]) -> str:
        lines = []
        for cid, mf in features.items():
            lines.append(f"[{cid}] mobility: {mf.mobility_summary}")
        return "\n".join(lines)
