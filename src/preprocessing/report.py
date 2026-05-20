import csv
import json
from datetime import datetime
from pathlib import Path

class RunReport:
    def __init__(self, out_dir: Path, name_prefix="run_report"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix
        self.rows = []
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    def add(self, row: dict):
        self.rows.append(dict(row))

    def write(self):
        csv_path = self.out_dir / f"{self.name_prefix}_{self.ts}.csv"
        json_path = self.out_dir / f"{self.name_prefix}_{self.ts}.json"

        # JSON
        json_path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")

        # CSV
        keys = sorted({k for r in self.rows for k in r.keys()})
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

        return str(csv_path), str(json_path)