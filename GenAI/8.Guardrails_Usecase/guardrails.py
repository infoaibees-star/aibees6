"""
PII guardrails: detect and redact personally identifiable information.

The whole rule set is compiled into a *single* alternation regex, so redacting a
response is one linear pass over the text instead of one pass per pattern.
Earlier rules win when two of them could match at the same position.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Match, NamedTuple, Sequence

# Month names, full or abbreviated (matched case-insensitively).
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"


class Rule(NamedTuple):
    """One PII pattern. Labels are free-form and may repeat across rules."""

    label: str
    pattern: str
    replacement: str


# Order matters: the first rule that matches at a given position wins, so the
# most specific pattern is listed first. ZIP+4 precedes SSN (both look like
# 5-then-4 digits), and street addresses precede bare ZIP codes (a house number
# can be five digits long).
RULES: tuple[Rule, ...] = (
    Rule("CreditCard", r"\b(?:\d{4}[\s\-]){3}\d{4}\b", "[CARD REDACTED]"),
    Rule("ZipCode", r"\b\d{5}-\d{4}\b", "[ZIP REDACTED]"),
    Rule("SSN", r"(?<!\d)\d{3}[\s\-]?\d{2}[\s\-]?\d{4}(?!\d)", "[SSN REDACTED]"),
    Rule(
        "Phone",
        r"(?:\(\s*\d{3}\s*\)\s*\d{3}[\s\-]?\d{4}|\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b)",
        "[PHONE REDACTED]",
    ),
    Rule("Email", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "[EMAIL REDACTED]"),
    Rule("AccountNumber", r"\bA\s*C\s*C\s*[\s\-]?\d{8,}\b", "[ACCOUNT REDACTED]"),
    Rule(
        "DateOfBirth",
        rf"(?:{_MONTH}\s+\d{{1,2}},?\s+\d{{4}}"          # March 14, 1965
        rf"|\d{{1,2}}\s+{_MONTH}\s+\d{{4}}"              # 14 March 1965
        r"|\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b)",        # 14/03/1965
        "[DOB REDACTED]",
    ),
    Rule(
        "StreetAddress",
        r"\b\d{1,5}\s+(?:[A-Za-z0-9#&'.\-]+\s){1,5}"
        r"(?:Street|St|Avenue|Ave|Drive|Dr|Road|Rd|Lane|Ln|"
        r"Boulevard|Blvd|Court|Ct|Place|Pl|Way|Circle|Cir)\b",
        "[ADDRESS REDACTED]",
    ),
    Rule("ZipCode", r"\b\d{5}\b", "[ZIP REDACTED]"),
)

# Keywords suggesting the *question* is fishing for PII.
_TRIGGER_KEYWORDS: tuple[str, ...] = (
    "ssn", "social security", "dob", "date of birth", "birth", "phone", "mobile",
    "cell", "email", "account", "address", "credit card", "card number", "zip", "postal",
)


def _compile(rules: Sequence[Rule]) -> tuple[re.Pattern[str], dict[str, Rule]]:
    """Fuse every rule into one alternation, keyed by generated group names.

    Group names are positional (`r0`, `r1`, …) rather than the labels themselves,
    so labels stay free-form and several rules may share one.
    """
    groups = {f"r{i}": rule for i, rule in enumerate(rules)}
    pattern = "|".join(f"(?P<{name}>{rule.pattern})" for name, rule in groups.items())
    return re.compile(pattern, re.IGNORECASE), groups


@dataclass(frozen=True)
class RedactionResult:
    """Outcome of a single :meth:`PIIGuardrail.redact` call."""

    original: str
    redacted: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def was_modified(self) -> bool:
        return bool(self.counts)

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def summary(self) -> str:
        if not self.counts:
            return "No PII detected."
        return "Redacted — " + ", ".join(f"{label}: {n}" for label, n in self.counts.items())


class PIIGuardrail:
    """Regex guardrail for LLM inputs (detection) and outputs (redaction).

    Extend it by appending a :class:`Rule` to :data:`RULES`, or by passing a custom
    rule set to the constructor — nothing else needs to change.
    """

    def __init__(self, rules: Iterable[Rule] = RULES) -> None:
        self._rules: tuple[Rule, ...] = tuple(rules)
        self._pattern, self._groups = _compile(self._rules)
        self._triggers = re.compile(
            r"\b(?:%s)\b" % "|".join(re.escape(kw) for kw in _TRIGGER_KEYWORDS),
            re.IGNORECASE,
        )

    @property
    def labels(self) -> tuple[str, ...]:
        """Distinct PII types this guardrail recognises, in rule order."""
        return tuple(dict.fromkeys(rule.label for rule in self._rules))

    def detect_pii_request(self, text: str) -> list[str]:
        """Return the PII-related keywords found in a user question (empty if none)."""
        if not text:
            return []
        # dict.fromkeys de-duplicates while preserving order of first appearance.
        return list(dict.fromkeys(m.group(0).lower() for m in self._triggers.finditer(text)))

    def redact(self, text: str) -> RedactionResult:
        """Replace every recognised PII value with its redaction token."""
        if not text:
            return RedactionResult(original=text or "", redacted=text or "")

        counts: dict[str, int] = {}

        def _replace(match: Match[str]) -> str:
            rule = self._groups[match.lastgroup or ""]
            counts[rule.label] = counts.get(rule.label, 0) + 1
            return rule.replacement

        return RedactionResult(
            original=text,
            redacted=self._pattern.sub(_replace, text),
            counts=counts,
        )
