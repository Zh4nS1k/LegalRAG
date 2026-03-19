from datetime import datetime, timedelta


class LegalCalculator:
    @staticmethod
    def calculate_penalty(base_amount, violation_type, circumstances):
        """Calculate penalties based on violation type and circumstances."""
        # Mechanical calculation rules
        multipliers = {"minor": 1.0, "moderate": 2.0, "severe": 5.0, "critical": 10.0}

        base_multiplier = multipliers.get(circumstances.get("severity", "minor"), 1.0)
        repeat_offender = circumstances.get("repeat_offender", False)
        if repeat_offender:
            base_multiplier *= 2.0

        total_penalty = base_amount * base_multiplier

        return {
            "base_amount": base_amount,
            "multiplier": base_multiplier,
            "total_penalty": total_penalty,
        }

    @staticmethod
    def calculate_deadline(start_date, days, business_days=False):
        """Calculate deadline from start date."""
        start = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if business_days:
            # Simple business day calculation (exclude weekends)
            current = start
            added_days = 0
            while added_days < days:
                current += timedelta(days=1)
                if current.weekday() < 5:  # Monday to Friday
                    added_days += 1
        else:
            current = start + timedelta(days=days)

        return current.isoformat()

    @staticmethod
    def validate_calculation_request(request):
        """Validate that calculation request has required fields."""
        required_fields = ["calculation_type", "parameters"]
        return all(field in request for field in required_fields)
