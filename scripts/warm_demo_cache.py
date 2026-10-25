"""Answer every example question once so the demo serves them from cache (run before export)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.service import EXAMPLES, get_service

svc = get_service()
for q in EXAMPLES:
    out = svc.ask(q)
    status = "cached" if out["cached"] else ("ERROR " + out["error"] if out["error"] else "answered")
    print(f"{out['timing']['total_s']:6.1f}s  {status:10.60s}  {q}")
