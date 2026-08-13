"""Append-only JSONL preprocessing manifest with failed-write rollback."""

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


PREPROCESSING_VERSION = "sar-fuse-tile-v2"


@dataclass(frozen=True)
class ManifestRecord:
    sample_id: str
    source_vv: str
    source_vh: str
    output_file: Optional[str]
    status: str
    reject_reason: Optional[str]
    preprocessing_version: str = PREPROCESSING_VERSION
    shape: Optional[Tuple[int, ...]] = None
    timestamp: str = ""
    checksum: Optional[str] = None

    def with_timestamp(self) -> "ManifestRecord":
        if self.timestamp:
            return self
        return ManifestRecord(
            **{
                **asdict(self),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )


class ManifestWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ManifestRecord) -> None:
        payload = asdict(record.with_timestamp())
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        previous_size = self.path.stat().st_size if self.path.exists() else 0
        try:
            with self.path.open("ab") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            if self.path.exists():
                with self.path.open("r+b") as handle:
                    handle.truncate(previous_size)
            raise
