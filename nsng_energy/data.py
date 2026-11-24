"""Pinned public data and manifest input for the benchmark."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from itertools import combinations
from pathlib import Path, PurePosixPath
import re
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

COMMIT = "858950c5bda24b3877e817983fa254f0f114d6a4"
UPSTREAM = "https://github.com/AlexisPapaioannou/Power-Consumption-Anomaly-Dataset"
ARCHIVE = f"{UPSTREAM}/archive/{COMMIT}.zip"
FAULTS = {"anomaly_Faulty_Thermostat", "anomaly_Inefficient_Heating_Elements"}
FIGURE_FILE = "anomaly_Faulty_Thermostat/dryer_1_day8_ANOMALIES.csv"


@dataclass
class Sequence:
    name: str
    group: str
    fault: str
    time: np.ndarray
    power: np.ndarray
    labels: np.ndarray
    voltage: np.ndarray | None = None
    normal_power: np.ndarray | None = None
    normal_key: str | None = None
    digest: str = ""


def prepare_public(destination: Path, archive: Path | None = None) -> Path:
    """Select a deterministic candidate using the aggregate constraints in Table I."""
    destination.mkdir(parents=True, exist_ok=True)
    if archive is None:
        with urlopen(ARCHIVE, timeout=90) as response:
            archive_bytes = response.read()
    else:
        archive_bytes = archive.read_bytes()
    with ZipFile(BytesIO(archive_bytes)) as source:
        files = {}
        for member in source.namelist():
            path = PurePosixPath(member)
            if (len(path.parts) == 6 and path.parts[1:4] == ("data", "Dryer", "Dryer_1")
                    and path.parts[4] in FAULTS | {"Normal"}
                    and re.fullmatch(r"dryer_1_day\d+(?:_ANOMALIES)?\.csv", path.name)):
                files["/".join(path.parts[4:])] = member
            if len(path.parts) == 2 and path.name == "LICENSE.txt":
                (destination / "LICENSE.txt").write_bytes(source.read(member))
        sizes = {}
        normal_dates = {}
        for key, member in sorted(files.items()):
            if key.split("/")[0] in FAULTS:
                sizes[key] = len(pd.read_csv(source.open(member), usecols=["label"]))
            elif key.startswith("Normal/"):
                data = pd.read_csv(source.open(member), usecols=["ctime"])
                normal_dates[key] = pd.to_datetime(data["ctime"], format="mixed")
        if len(sizes) != 20 or len(normal_dates) != 10:
            raise ValueError("The public archive must contain both dryer faults and 10 normal days.")
        matching = [
            subset for subset in combinations(sizes, 15)
            if FIGURE_FILE in subset
            and sum(sizes[name] for name in subset) == 22380
            and min(sizes[name] for name in subset) == 1470
            and max(sizes[name] for name in subset) == 1550
        ]
        if not matching:
            raise ValueError("No subset of the archive matches the published aggregate constraints.")
        # Keep overlapping acquisition windows in the same fold.
        groups = {}
        end = pd.Timestamp.min
        for name, dates in sorted(normal_dates.items(), key=lambda item: item[1].min()):
            if dates.min() > end:
                group = dates.min().date().isoformat()
            groups[name] = group
            end = max(end, dates.max())
        rows = []
        for index, key in enumerate(matching[0], 1):
            day = re.search(r"day(\d+)", key).group(1)
            normal = f"Normal/dryer_1_day{day}.csv"
            for name in (key, normal):
                target = destination / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(source.read(files[name]))
            rows.append({
                "sequence": f"dryer_{index:02d}", "path": key, "group": groups[normal],
                "fault": key.split("/")[0].removeprefix("anomaly_"),
                "normal_path": normal, "figure": int(key == FIGURE_FILE),
            })
    manifest = destination / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    pd.Series({
        "upstream": UPSTREAM, "commit": COMMIT, "archive_sha256": sha256(archive_bytes).hexdigest(),
        "matching_subsets_including_figure": len(matching),
        "selection": "lexicographically first relative-path tuple",
    }).to_json(destination / "source.json", indent=2)
    return manifest


def _read_series(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if not {"ctime", "activePower"} <= set(frame):
        raise ValueError(f"{path} requires ctime and activePower columns.")
    if frame.empty or not np.isfinite(frame["activePower"].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} requires finite power observations.")
    times = pd.to_datetime(frame["ctime"], format="mixed", errors="raise")
    if times.isna().any() or not times.is_monotonic_increasing:
        raise ValueError(f"{path} requires nondecreasing timestamps in recorded row order.")
    return frame


def load_manifest(path: Path) -> tuple[list[Sequence], str]:
    manifest = pd.read_csv(path, keep_default_na=False, dtype=str)
    required = {"sequence", "path", "group"}
    if not required <= set(manifest) or manifest.empty:
        raise ValueError(f"{path} requires sequence, path and group columns and at least one row.")
    if manifest[list(required)].eq("").any().any() or not manifest["sequence"].is_unique:
        raise ValueError("Sequence names must be unique and all required manifest fields nonempty.")
    if "figure" in manifest and manifest["figure"].eq("1").sum() > 1:
        raise ValueError("Select at most one figure sequence.")
    assigned_groups = {}
    normal_power = {}
    sequences = []
    figure = manifest["sequence"].iloc[0]
    for row in manifest.to_dict("records"):
        file = (path.parent / row["path"]).resolve()
        if file in assigned_groups:
            raise ValueError(f"Repeated evaluation path in manifest: {file}")
        data = _read_series(file)
        if "label" not in data or not data["label"].isin([0, 1]).all():
            raise ValueError(f"{file} requires binary labels (0 or 1).")
        voltage = data["voltage"].to_numpy(dtype=float) if "voltage" in data else None
        if voltage is not None and (np.isinf(voltage).any() or np.any(voltage[np.isfinite(voltage)] <= 0)):
            raise ValueError(f"{file} requires positive voltage measurements or empty values.")
        normal_key = row.get("normal_path")
        if normal_key:
            normal_file = (path.parent / normal_key).resolve()
            if normal_file not in normal_power:
                normal = _read_series(normal_file)
                if "label" in normal and not normal["label"].eq(0).all():
                    raise ValueError(f"{normal_file} must contain normal observations only.")
                normal_power[normal_file] = normal["activePower"].to_numpy(dtype=float)
            previous_group = assigned_groups.setdefault(normal_file, row["group"])
            if previous_group != row["group"]:
                raise ValueError(f"{normal_file} is shared by different groups.")
        assigned_groups[file] = row["group"]
        sequences.append(Sequence(
            row["sequence"], row["group"], row.get("fault", ""),
            data["ctime"].to_numpy(dtype=str), data["activePower"].to_numpy(dtype=float),
            data["label"].to_numpy(dtype=np.int64), voltage,
            normal_power[normal_file] if normal_key else None, normal_key,
            sha256(file.read_bytes()).hexdigest(),
        ))
        if row.get("figure") == "1":
            figure = row["sequence"]
    return sequences, figure
