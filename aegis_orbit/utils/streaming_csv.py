"""CSV writer that emits rows as they are produced."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class StreamingCSVWriter:
    """Write CSV rows incrementally without replacing an existing file."""

    def __init__(self, path: Path, fieldnames: list[str] | tuple[str, ...]) -> None:
        self.path = path
        self._handle = path.open("x", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=fieldnames,
            extrasaction="raise",
        )
        self._writer.writeheader()
        self.flush()

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)

    def flush(self) -> None:
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> StreamingCSVWriter:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
