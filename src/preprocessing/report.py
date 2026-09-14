import csv
import fcntl
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

class RunReport:
    def __init__(self, out_dir: Path, name_prefix="run_report"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix
        self.rows = []
        self.ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.job_id = f"{self.ts}_{os.getpid()}_{uuid.uuid4().hex[:8]}"

    def add(self, row: dict):
        self.rows.append(dict(row))

    @staticmethod
    def _atomic_write_text(path: Path, text: str) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    @classmethod
    def _write_csv_atomic(cls, path: Path, rows: list[dict]) -> None:
        keys = sorted({k for row in rows for k in row.keys()})
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=keys)
            if keys:
                writer.writeheader()
                writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def _read_all_job_rows(self) -> list[dict]:
        merged_by_scene: dict[str, dict] = {}
        rows_without_scene: list[dict] = []
        for path in sorted(self.out_dir.glob(f"{self.name_prefix}_*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, list):
                continue
            for row in payload:
                if not isinstance(row, dict):
                    continue
                scene = row.get("scene")
                if scene:
                    merged_by_scene[str(scene)] = row
                else:
                    rows_without_scene.append(row)
        return [merged_by_scene[key] for key in sorted(merged_by_scene)] + rows_without_scene

    def write(self):
        # Preserve one immutable report per parallel job.
        job_json_path = self.out_dir / f"{self.name_prefix}_{self.job_id}.json"
        job_csv_path = self.out_dir / f"{self.name_prefix}_{self.job_id}.csv"
        self._atomic_write_text(job_json_path, json.dumps(self.rows, indent=2))
        self._write_csv_atomic(job_csv_path, self.rows)

        # Lock merged updates and replace reports atomically.
        merged_json_path = self.out_dir / f"{self.name_prefix}.json"
        merged_csv_path = self.out_dir / f"{self.name_prefix}.csv"
        lock_path = self.out_dir / f".{self.name_prefix}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                merged_rows = self._read_all_job_rows()
                self._atomic_write_text(merged_json_path, json.dumps(merged_rows, indent=2))
                self._write_csv_atomic(merged_csv_path, merged_rows)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        return str(merged_csv_path), str(merged_json_path)
