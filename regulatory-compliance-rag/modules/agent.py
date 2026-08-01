from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from uuid import uuid4

from modules.models import Citation, StrategyCheckResponse


# Map the main strategy risk categories to their keyword triggers and severity.
RISK_RULES: list[tuple[str, list[str], str, str]] = [
	(
		"insider trading and market abuse",
		["insider", "inside information", "material nonpublic", "front running", "market abuse"],
		"The strategy suggests possible insider trading or market abuse.",
		"BLOCKED",
	),
	(
		"personal data misuse",
		["personal data", "customer data", "gdpr", "pii", "profile", "consent"],
		"The strategy appears to process personal data in a way that may conflict with GDPR controls.",
		"NEEDS_REVIEW",
	),
	(
		"unsuitable execution behavior",
		["best execution", "suitability", "market manipulation", "spoofing", "layering"],
		"The strategy may conflict with best execution, suitability, or market integrity obligations.",
		"NEEDS_REVIEW",
	),
]


@dataclass(frozen=True)
class ComplianceDecision:
	decision: str
	reason: str
	citations: list[Citation]


class ComplianceAgent:
	# Evaluate a strategy against retrieved regulations and basic risk signals.
	def evaluate(self, strategy: str, retrieved_rules: Iterable[dict], risk_profile: dict[str, object] | None = None) -> StrategyCheckResponse:
		retrieved_rules = list(retrieved_rules)
		risk_profile = risk_profile or {}
		strategy_lower = strategy.lower()
		strategy_id = f"strategy-{uuid4().hex[:12]}"
		timestamp = datetime.now(timezone.utc).isoformat()
		citations = self._build_citations(retrieved_rules, limit=5)
		confidence = self._confidence_score(strategy_lower, citations)
		risk_flags: list[str] = []
		recommendations: list[str] = []
		final_decision = "APPROVED"
		reason = "APPROVED: No high-confidence regulatory conflict was detected in the retrieved policy set."
		matched_rules: list[tuple[str, str, str]] = []

		for rule_name, keywords, blocked_reason, severity in RISK_RULES:
			if any(keyword in strategy_lower for keyword in keywords):
				matched_rules.append((rule_name, blocked_reason, severity))

		if matched_rules:
			risk_flags.extend([rule_name for rule_name, _, _ in matched_rules])
			if any(severity == "BLOCKED" for _, _, severity in matched_rules):
				blocked_rule = next(item for item in matched_rules if item[2] == "BLOCKED")
				final_decision = "BLOCKED"
				reason = f"BLOCKED: {blocked_rule[1]}"
				recommendations.append("Review the strategy against the cited regulations before execution.")
			else:
				review_rule = matched_rules[0]
				final_decision = "NEEDS_REVIEW"
				reason = f"NEEDS_REVIEW: {review_rule[1]}"
				recommendations.append("Clarify the strategy details and confirm the applicable compliance requirements.")

		if final_decision == "APPROVED" and bool(risk_profile.get("use_insider_info")):
			final_decision = "BLOCKED"
			reason = "BLOCKED: Strategies that use insider information are not permitted."
			risk_flags.append("insider_information")
			recommendations.append("Remove any dependency on material nonpublic information.")

		if final_decision == "APPROVED":
			leverage_ratio = risk_profile.get("leverage_ratio")
			if isinstance(leverage_ratio, (int, float)) and leverage_ratio > 2:
				final_decision = "NEEDS_REVIEW"
				reason = "NEEDS_REVIEW: Leverage above 2x needs human review against desk and jurisdiction-specific rules."
				risk_flags.append("high_leverage")
				recommendations.append("Confirm leverage permissions for the target jurisdiction before approval.")

		if final_decision == "APPROVED" and bool(risk_profile.get("short_selling")):
			if any("short" in rule["text"].lower() for rule in retrieved_rules):
				final_decision = "NEEDS_REVIEW"
				reason = "NEEDS_REVIEW: Short-selling may require jurisdiction-specific checks and position-limit review."
				risk_flags.append("short_selling_review")
				recommendations.append("Verify short-selling constraints before execution.")

		if final_decision == "APPROVED" and self._is_ambiguous(strategy_lower, retrieved_rules):
			final_decision = "NEEDS_REVIEW"
			reason = "NEEDS_REVIEW: The strategy is borderline or missing enough detail for a confident automated decision."
			risk_flags.append("ambiguous_strategy")
			recommendations.append("Provide more detail on leverage, data usage, short-selling, and jurisdiction before approval.")

		if final_decision == "BLOCKED" and citations:
			reason = f"{reason} Relevant regulations were identified in the retrieved policy set."

		if final_decision == "NEEDS_REVIEW" and not citations:
			recommendations.append("Upload or index more applicable regulations for a stronger compliance signal.")

		return StrategyCheckResponse(
			strategy_id=strategy_id,
			decision=final_decision,
			confidence=confidence,
			reason=reason,
			citations=citations,
			retrieved_count=len(retrieved_rules),
			risk_flags=risk_flags,
			recommendations=recommendations,
			timestamp=timestamp,
		)

	# Convert retrieved rules into the citation objects returned by the API.
	@staticmethod
	def _build_citations(retrieved_rules: list[dict], limit: int) -> list[Citation]:
		citations: list[Citation] = []
		for rule in retrieved_rules[:limit]:
			citations.append(
				Citation(
					source_id=rule["source_id"],
					source_name=rule["source_name"],
					regulation_title=rule["regulation_title"],
					jurisdiction=rule["jurisdiction"],
					effective_date=rule["effective_date"],
					section=rule["section"],
					chunk_id=rule["chunk_id"],
					confidence=rule.get("confidence", rule.get("bm25_score", 0.0)),
					excerpt=rule["text"][:240],
					retrieval_score=rule.get("retrieval_score", rule.get("bm25_score", 0.0)),
				)
			)
		return citations

	# Decide whether the strategy is too vague to score confidently.
	@staticmethod
	def _is_ambiguous(strategy: str, retrieved_rules: list[dict]) -> bool:
		ambiguous_markers = ["maybe", "could", "potentially", "unclear", "borderline", "experimental", "highly leveraged"]
		if any(marker in strategy for marker in ambiguous_markers):
			return True
		if not retrieved_rules:
			return True
		return max(rule.get("confidence", rule.get("bm25_score", 0.0)) for rule in retrieved_rules) < 0.3

	# Turn retrieval confidence into a bounded score for the response payload.
	@staticmethod
	def _confidence_score(strategy: str, citations: list[Citation]) -> float:
		if not citations:
			return 0.35
		base = max(citation.confidence for citation in citations)
		if any(flag in strategy for flag in ["insider", "market abuse", "front running", "gdpr", "personal data"]):
			return min(0.99, base + 0.08)
		return min(0.95, base + 0.05)









