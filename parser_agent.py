"""parser_agent.py — Sarvam OCR utility (not called in inference loop)."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

def parse_document(file_path: str) -> list[dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r'\b(clause|section|article)\s+(\d+[\.\d]*)[:\s]+(.{20,500}?)(?=\b(?:clause|section|article)\s+\d|$)',
        re.IGNORECASE | re.DOTALL
    )
    clauses = []
    for i, m in enumerate(pattern.finditer(text), 1):
        clauses.append({"id": f"C{i:03d}", "type": m.group(1).lower(),
                        "number": m.group(2), "text": m.group(3).strip(),
                        "source_file": path.name})
    if not clauses:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 30]
        clauses = [{"id": f"C{i:03d}", "text": s, "source_file": path.name}
                   for i, s in enumerate(sentences[:20], 1)]
    return clauses

if __name__ == "__main__":
    print("parser_agent imports OK")
