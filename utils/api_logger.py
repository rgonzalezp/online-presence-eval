# api_logger.py

import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(__file__).parent / "api_calls.json"

def _serialize(obj):
    """Recursively turn obj into JSON-serializable primitives or repr(obj)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    try:
        if hasattr(obj, "to_dict"):
            return _serialize(obj.to_dict())
        if hasattr(obj, "json"):
            return _serialize(obj.json())
    except Exception:
        pass
    return repr(obj)

def log_api_call(
    model: str,
    mode: str,
    job_variant: str,
    name_variant: str,
    prompt: str,
    response_text: str,
    metadata: dict,
    raw_response: object,
    history: list
):
    """Append one record to a JSON array on disk."""
    record = {
        "timestamp":     datetime.utcnow().isoformat(),
        "model":         model,
        "mode":          mode,
        "job_variant":   job_variant,
        "name_variant":  name_variant,
        "prompt":        prompt,
        "response_text": response_text,
        "metadata":      _serialize(metadata),
        "raw_response":  _serialize(raw_response),
        "history":       _serialize(history),
    }

    # load existing array (or start fresh)
    if LOG_PATH.exists():
        try:
            data = json.loads(LOG_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    # append & write back
    data.append(record)
    LOG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
