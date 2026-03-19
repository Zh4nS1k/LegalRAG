# confidence_calculator.py — Level 2: Scripts (Hard Logic)
# Mathematically calculate ConfidenceScore from similarity_score (Pinecone).

import math


class ConfidenceCalculator:
    @staticmethod
    def calculate_confidence(similarity_scores, threshold=0.5):
        """Calculate confidence from top similarity scores.
        Formula: max_score * (1 + log(1 + avg_top_3)) / 2
        """
        if not similarity_scores:
            return 0.0

        scores = sorted(similarity_scores, reverse=True)
        max_score = scores[0]
        top_3_avg = sum(scores[:3]) / min(3, len(scores))

        # Boost for consistent high scores
        consistency_bonus = math.log(1 + top_3_avg)

        confidence = (max_score + consistency_bonus) / 2.0
        return min(1.0, max(0.0, confidence))

    @staticmethod
    def is_above_threshold(confidence, threshold=0.7):
        """Check if confidence meets threshold."""
        return confidence >= threshold
