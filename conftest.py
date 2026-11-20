"""
Root-level conftest.py — patches json.JSONEncoder to handle datetime objects
so tests that call json.dumps on fixtures containing datetime values work correctly.
"""

import json
from datetime import date, datetime

_original_default = json.JSONEncoder.default


def _patched_default(self, obj):
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return _original_default(self, obj)


json.JSONEncoder.default = _patched_default
