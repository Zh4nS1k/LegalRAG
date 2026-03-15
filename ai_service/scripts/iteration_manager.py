# iteration_manager.py — Level 2: Scripts (Hard Logic)
# Counts clarification iterations. If counter >= 2, force synthesis=True.

class IterationController:
    def __init__(self, max_iterations=2):
        self.max_iterations = max_iterations

    def count_clarifications(self, history):
        """Count how many times user was asked for clarification."""
        if not history:
            return 0
        count = 0
        for msg in history:
            if msg.get("role") == "assistant":
                content = msg.get("content", "").lower()
                if "уточн" in content or "clarify" in content or "поясн" in content:
                    count += 1
        return count

    def should_force_synthesis(self, history):
        """If >= max_iterations clarifications, force synthesis."""
        return self.count_clarifications(history) >= self.max_iterations

    def modify_request(self, request, history):
        """Modify request to force synthesis if needed."""
        if self.should_force_synthesis(history):
            request["force_synthesis"] = True
        return request
