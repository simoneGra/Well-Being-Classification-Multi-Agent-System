"""
DataLoaderAgent: Loads and normalizes all data sources for a given dataset directory.
No LLM calls - pure Python data loading.
"""
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class StatusEvent:
    event_id: int
    citizen_id: str
    event_type: str
    physical_activity_index: float
    sleep_quality_index: float
    environmental_exposure_level: float
    timestamp: str


@dataclass
class LocationRecord:
    user_id: str
    timestamp: str
    lat: float
    lng: float
    city: str


@dataclass
class UserProfile:
    user_id: str
    first_name: str
    last_name: str
    birth_year: int
    job: str
    city: str
    lat: float
    lng: float


@dataclass
class DataBundle:
    """All data for one dataset (train or eval)."""
    events: List[StatusEvent] = field(default_factory=list)
    locations: List[LocationRecord] = field(default_factory=list)
    users: Dict[str, UserProfile] = field(default_factory=dict)
    personas: str = ""
    citizen_ids: List[str] = field(default_factory=list)


class DataLoaderAgent:
    """Loads all data sources from a directory and returns a DataBundle."""

    def run(self, data_dir: Path) -> DataBundle:
        bundle = DataBundle()
        bundle.events = self._load_status(data_dir / "status.csv")
        bundle.locations = self._load_locations(data_dir / "locations.json")
        bundle.users = self._load_users(data_dir / "users.json")
        bundle.personas = self._load_personas(data_dir / "personas.md")

        seen = []
        for ev in bundle.events:
            if ev.citizen_id not in seen:
                seen.append(ev.citizen_id)
        bundle.citizen_ids = seen

        return bundle

    def _load_status(self, path: Path) -> List[StatusEvent]:
        events = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row["EventID"].strip():
                    continue
                events.append(StatusEvent(
                    event_id=int(row["EventID"]),
                    citizen_id=row["CitizenID"].strip(),
                    event_type=row["EventType"].strip().lower(),
                    physical_activity_index=float(row["PhysicalActivityIndex"]),
                    sleep_quality_index=float(row["SleepQualityIndex"]),
                    environmental_exposure_level=float(row["EnvironmentalExposureLevel"]),
                    timestamp=row["Timestamp"].strip(),
                ))
        return events

    def _load_locations(self, path: Path) -> List[LocationRecord]:
        with open(path) as f:
            raw = json.load(f)
        records = []
        for item in raw:
            uid = item.get("user_id") or item.get("BioTag", "")
            records.append(LocationRecord(
                user_id=uid,
                timestamp=str(item.get("timestamp", "")),
                lat=float(item.get("lat", 0)),
                lng=float(item.get("lng", 0)),
                city=str(item.get("city", "")),
            ))
        return records

    def _load_users(self, path: Path) -> Dict[str, UserProfile]:
        with open(path) as f:
            raw = json.load(f)
        users = {}
        for item in raw:
            uid = item["user_id"]
            res = item.get("residence", {})
            users[uid] = UserProfile(
                user_id=uid,
                first_name=item.get("first_name", ""),
                last_name=item.get("last_name", ""),
                birth_year=int(item.get("birth_year", 0)),
                job=item.get("job", ""),
                city=res.get("city", ""),
                lat=float(res.get("lat", 0)),
                lng=float(res.get("lng", 0)),
            )
        return users

    def _load_personas(self, path: Path) -> str:
        with open(path) as f:
            return f.read()
