# type_sanitizer.py — Level 2: Scripts (Hard Logic)
# Clean numpy types from JSON before sending to Go backend.

import json

class TypeGuard:
    @staticmethod
    def clean_for_json(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: TypeGuard.clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [TypeGuard.clean_for_json(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj

    @staticmethod
    def safe_json_dumps(data):
        """Safe JSON dump with type cleaning."""
        cleaned = TypeGuard.clean_for_json(data)
        return json.dumps(cleaned, ensure_ascii=False, default=str)
