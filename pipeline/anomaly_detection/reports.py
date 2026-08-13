from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class InferenceRunReport:
    expected_tiles: int = 0
    processed_tiles: int = 0
    failed_tiles: int = 0
    missing_pairs: int = 0
    output_tiles: int = 0
    failed_files: List[str] = field(default_factory=list)
    fatal_errors: List[str] = field(default_factory=list)
    detector_metadata: Dict[str, str] = field(default_factory=dict)
    output_artifact: str = ""

    @property
    def status(self) -> str:
        if self.fatal_errors or self.expected_tiles == 0 or self.processed_tiles == 0:
            return "FAILED"
        if self.failed_tiles or self.missing_pairs or self.processed_tiles < self.expected_tiles:
            return "PARTIAL"
        return "SUCCESS"

    def to_dict(self) -> Dict[str, Any]:
        return {"status": self.status, **asdict(self)}
