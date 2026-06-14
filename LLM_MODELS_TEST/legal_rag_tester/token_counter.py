"""Token counting utility using tiktoken."""
import tiktoken
from collections import defaultdict
from typing import Dict

class TokenCounter:
    """Estimates tokens using tiktoken cl100k_base."""
    
    def __init__(self):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.session_totals: Dict[str, int] = defaultdict(int)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))

    def count_prompt(self, system: str, user: str) -> int:
        return self.count(system) + self.count(user) + 8

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        self.session_totals[model] += prompt_tokens + completion_tokens

    def session_summary(self) -> Dict[str, int]:
        totals = dict(self.session_totals)
        totals["TOTAL"] = sum(self.session_totals.values())
        return totals