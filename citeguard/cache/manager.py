from __future__ import annotations
import json
from pathlib import Path
from citeguard.models import VerificationResult


class CacheManager:
    """Manages JSON checkpoint files and PDF index cache."""

    def __init__(self, checkpoints_dir: Path, index_cache_dir: Path) -> None:
        self._ckpt_dir = Path(checkpoints_dir)
        self._idx_dir = Path(index_cache_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._idx_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, result: VerificationResult) -> None:
        path = self._ckpt_dir / f"{result.citation_id}.json"
        path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    def load_checkpoint(self, citation_id: str) -> VerificationResult | None:
        path = self._ckpt_dir / f"{citation_id}.json"
        if not path.exists():
            return None
        try:
            return VerificationResult.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def completed_citation_ids(self) -> set[str]:
        return {p.stem for p in self._ckpt_dir.glob("*.json")}

    def all_results(self) -> list[VerificationResult]:
        results = []
        for path in sorted(self._ckpt_dir.glob("*.json")):
            r = self.load_checkpoint(path.stem)
            if r is not None:
                results.append(r)
        return results

    def index_cache_path(self, pdf_path: str) -> Path:
        safe_name = Path(pdf_path).stem.replace(" ", "_")
        return self._idx_dir / f"{safe_name}.pkl"

    def index_cache_exists(self, pdf_path: str) -> bool:
        return self.index_cache_path(pdf_path).exists()
