from typing import Any, Dict, List


class EvaluationResult:
    def __init__(self, attempts: int, strategy_used: str, confidence: float, result_count: int):
        self.attempts = attempts
        self.strategy_used = strategy_used
        self.confidence = confidence
        self.result_count = result_count


class EvaluationEngine:
    """Simple evaluator based on search scores and QA confidence."""

    def evaluate(self, query: str, strategy: str, results: List[Dict[str, Any]]) -> EvaluationResult:
        if not results:
            return EvaluationResult(attempts=1, strategy_used=strategy, confidence=0.0, result_count=0)

        scores = []
        confidences = []
        for r in results:
            s = r.get("score")
            if isinstance(s, (int, float)):
                scores.append(float(s))
            c = r.get("qa_confidence")
            if isinstance(c, (int, float)):
                confidences.append(float(c))

        avg_score = sum(scores) / len(scores) if scores else 0.0
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        confidence = max(avg_score, avg_conf)
        return EvaluationResult(attempts=1, strategy_used=strategy, confidence=confidence, result_count=len(results))