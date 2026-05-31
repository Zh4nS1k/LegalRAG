# confidence_calculator.py — Level 2: Scripts (Hard Logic)
# Heuristic confidence scoring for similarity outputs.


class ConfidenceCalculator:
    @staticmethod
    def calculate_confidence(similarity_scores, threshold=0.5):
        """Calculate confidence from top similarity scores.
        Heuristic only: linear blend of best score and top-3 average.
        Calibrate threshold offline on a labeled benchmark set.
        """
        if not similarity_scores:
            return 0.0

        scores = sorted(similarity_scores, reverse=True)
        max_score = scores[0]
        top_3_avg = sum(scores[:3]) / min(3, len(scores))
        confidence = (0.7 * max_score) + (0.3 * top_3_avg)
        return round(min(1.0, max(0.0, confidence)), 3)

    @staticmethod
    def is_above_threshold(confidence, threshold=0.65):
        """Check if confidence meets threshold."""
        return confidence >= threshold
