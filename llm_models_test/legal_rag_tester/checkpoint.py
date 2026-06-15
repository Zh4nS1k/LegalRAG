"""Crash-safe checkpoint manager for the Legal RAG pipeline."""
from pathlib import Path
from typing import List
import excel_io
from models.schemas import TestRow
from logger import pipeline_logger


class CheckpointManager:
    """
    Writes results to disk after EVERY completed question.
    On any crash, the partial file is always valid and readable.
    """

    def __init__(self, output_dir: str, session_id: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        # Live checkpoint — overwritten atomically after each question
        self.checkpoint_path = self.output_dir / f"checkpoint_{session_id}.xlsx"
        # Final file — written only on clean completion
        self.final_path = self.output_dir / f"results_{session_id}.xlsx"
        self.rows: List[TestRow] = []

    def add_question_results(self, results: List[TestRow]) -> None:
        """Call after each question completes. Saves immediately to disk."""
        self.rows.extend(results)
        self._write(self.checkpoint_path)
        pipeline_logger.log_simple_info(
            f"💾 Checkpoint: {len(self.rows)} rows → {self.checkpoint_path.name}"
        )

    def finalize(self) -> Path:
        """Call on clean completion. Writes final named results file."""
        if not self.rows:
            pipeline_logger.log_warning("No rows to finalize.")
            return self.final_path
        self._write(self.final_path)
        pipeline_logger.log_simple_success(
            f"✅ Final results saved → {self.final_path.name}"
        )
        return self.final_path

    def _write(self, path: Path) -> None:
        """Write all rows to Excel atomically via a temp file."""
        if not self.rows:
            return
        tmp_path = path.with_suffix(".tmp.xlsx")
        excel_io.write_results(self.rows, output_path=tmp_path)
        tmp_path.replace(path)  # atomic rename — no corrupt file on crash

    @staticmethod
    def find_latest_checkpoint(output_dir: str) -> "Path | None":
        """Find the most recent checkpoint file to resume from."""
        checkpoints = sorted(
            Path(output_dir).glob("checkpoint_*.xlsx"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        return checkpoints[0] if checkpoints else None
