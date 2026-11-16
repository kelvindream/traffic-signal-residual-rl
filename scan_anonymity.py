#!/usr/bin/env python3
"""Scan the anonymous package for prohibited artifacts and common identifiers.

Project-specific author identifiers are intentionally supplied only at scan time
through ANONYMITY_FORBIDDEN_TERMS (pipe-separated). They are not stored in the
anonymous package itself.
"""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
TEXT_SUFFIXES = {
    ".py", ".txt", ".md", ".csv", ".json", ".yaml", ".yml",
    ".xml", ".cfg", ".ini", ".sh",
}
GENERIC_FORBIDDEN_TEXT = (
    "/home/",
    "@qq.com",
)
FORBIDDEN_PATH_PARTS = (
    "main.tex", "cover letter", "cover_letter",
    "suggested reviewer", "suggested_reviewer",
)
FORBIDDEN_BINARY_SUFFIXES = (".pt", ".pth", ".ckpt", ".pyc")

extra_terms = tuple(
    term.strip()
    for term in os.environ.get("ANONYMITY_FORBIDDEN_TERMS", "").split("|")
    if term.strip()
)
forbidden_text = GENERIC_FORBIDDEN_TEXT + extra_terms

hits = []
for path in sorted(ROOT.rglob("*")):
    if path.resolve() == Path(__file__).resolve():
        continue
    rel = path.relative_to(ROOT).as_posix()
    lower = rel.lower()
    if any(token in lower for token in FORBIDDEN_PATH_PARTS):
        hits.append(f"prohibited name: {rel}")
    if path.is_file() and path.suffix.lower() in FORBIDDEN_BINARY_SUFFIXES:
        hits.append(f"prohibited binary/cache file: {rel}")
    if path.is_dir() and path.name.lower() in {"checkpoint", "checkpoints", "__pycache__"}:
        hits.append(f"prohibited directory: {rel}")
    if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
        continue
    text = path.read_text(encoding="utf-8", errors="replace")
    for token in forbidden_text:
        if token.lower() in text.lower():
            hits.append(f"prohibited text {token!r}: {rel}")

if hits:
    print("ANONYMITY SCAN FAILED")
    print("\n".join(hits))
    sys.exit(1)
print("ANONYMITY SCAN PASSED")
