# tanml/ui/services/validation.py
"""
Validation execution service for TanML UI.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from tanml.engine.core_engine_agent import ValidationEngine


def _try_run_engine(
    engine: ValidationEngine, progress_cb: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    try:
        return engine.run_all_checks(progress_callback=progress_cb)
    except TypeError:
        return engine.run_all_checks()
