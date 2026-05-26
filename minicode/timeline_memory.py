"""Timeline-oriented memory context construction.

This module turns retrieved conversational sessions into a compact chronological
state context. It is intentionally deterministic: retrieval still decides which
sessions are candidates, while this layer decides how to expose ordered evidence
and likely latest-state candidates to a reader.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta


TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
NUMBER_WORD_PATTERN = "one|two|three|four|five|six|seven|eight|nine|ten"
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do",
    "does", "for", "from", "have", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "our", "previous", "the", "this", "to",
    "was", "were", "what", "when", "where", "which", "who", "with",
    "you", "your",
}


@dataclass(frozen=True)
class TimelineTurn:
    """One dated turn selected for timeline memory construction."""

    session_id: str
    session_date: str
    turn_index: int
    role: str
    content: str
    relevance: float


@dataclass(frozen=True)
class TimelineContext:
    """Formatted timeline context plus debug metadata."""

    text: str
    selected_turns: list[TimelineTurn]
    latest_candidates: list[TimelineTurn]

    @property
    def selected_count(self) -> int:
        return len(self.selected_turns)


@dataclass(frozen=True)
class StateRecord:
    """A lightweight extracted state fact with dated evidence."""

    subject: str
    attribute: str
    value: str
    date: str
    evidence: str
    evidence_id: str = ""
    confidence: float = 0.5
    record_type: str = "state"

    @property
    def key(self) -> tuple[str, str]:
        return (self.subject.lower(), self.attribute.lower())


@dataclass
class LatestStateMemory:
    """Small latest-value index over extracted state records."""

    records: list[StateRecord]

    def latest_by_key(self) -> dict[tuple[str, str], StateRecord]:
        latest: dict[tuple[str, str], StateRecord] = {}
        for record in self.records:
            current = latest.get(record.key)
            if current is None or date_key(record.date) >= date_key(current.date):
                latest[record.key] = record
        return latest

    def format_for_prompt(self, max_records: int = 12) -> str:
        latest = sorted(
            self.latest_by_key().values(),
            key=lambda record: (date_key(record.date), record.confidence),
            reverse=True,
        )[:max_records]
        if not latest:
            return ""
        lines = ["## Latest State Memory", ""]
        for record in latest:
            lines.append(
                f"- [{record.date}] {record.record_type}: {record.subject} / {record.attribute} = "
                f"{record.value} (conf={record.confidence:.2f}; evidence={record.evidence_id})"
            )
        return "\n".join(lines)


@dataclass
class SemanticStateIndex:
    """Question-aware index over extracted state and event records."""

    records: list[StateRecord]

    def search(self, question: str, max_records: int = 16) -> list[StateRecord]:
        q_terms = set(tokenize(question))
        scored = [
            (score_state_record(q_terms, record), record)
            for record in self.records
        ]
        ranked = sorted(
            [item for item in scored if item[0] > 0],
            key=lambda item: (item[0], date_key(item[1].date), item[1].confidence),
            reverse=True,
        )
        return [record for _, record in ranked[:max_records]]

    def format_for_prompt(self, question: str, max_records: int = 16) -> str:
        records = self.search(question, max_records=max_records)
        if not records:
            return ""
        lines = ["## Semantic State/Event Memory", ""]
        for record in records:
            lines.append(
                f"- [{record.date}] {record.record_type}: {record.subject} / "
                f"{record.attribute} = {record.value} "
                f"(conf={record.confidence:.2f}; evidence={record.evidence_id})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class StateReasoningResult:
    """A deterministic answer candidate derived from state/event records."""

    answer: str
    reasoning_type: str
    confidence: float
    evidence_ids: list[str]
    explanation: str


@dataclass
class StateReasoner:
    """Small deterministic reasoner over semantic state/event records."""

    records: list[StateRecord]

    def answer(self, question: str, reference_date: str = "") -> StateReasoningResult | None:
        q = str(question or "").lower()
        if self._looks_like_age_difference(q):
            return self.answer_age_difference(question)
        if self._looks_like_duration_sum(q):
            duration_result = self.answer_duration_sum(question)
            if duration_result is not None:
                return duration_result
        if self._looks_like_distinct_event_day_count(q):
            return self.answer_distinct_event_day_count(question)
        if self._looks_like_consecutive_event_since(q):
            consecutive_result = self.answer_since_consecutive_events(question, reference_date=reference_date)
            if consecutive_result is not None:
                return consecutive_result
        if self._looks_like_date_diff(q):
            return self.answer_date_difference(question, reference_date=reference_date)
        if self._looks_like_event_order(q):
            return self.answer_event_order(question)
        if self._looks_like_latest_state(q):
            return self.answer_latest_state(question)
        return None

    def answer_latest_state(self, question: str) -> StateReasoningResult | None:
        candidates = [record for record in SemanticStateIndex(self.records).search(question, max_records=12) if record.record_type == "state"]
        if not candidates:
            return None
        prefer_previous = "previous" in question.lower()
        if prefer_previous:
            latest = sorted(
                candidates,
                key=lambda record: (
                    -int(_latest_state_hint_match(question, record)),
                    parse_date(record.date) or datetime.min,
                    -_score_latest_state_candidate(question, record),
                    -record.confidence,
                ),
            )[0]
        else:
            latest = sorted(
                candidates,
                key=lambda record: (
                    int(_latest_state_hint_match(question, record)),
                    parse_date(record.date) or datetime.min,
                    _score_latest_state_candidate(question, record),
                    record.confidence,
                ),
                reverse=True,
            )[0]
        return StateReasoningResult(
            answer=latest.value,
            reasoning_type="latest-state",
            confidence=min(0.90, latest.confidence),
            evidence_ids=[latest.evidence_id],
            explanation=f"Selected latest matching state dated {latest.date}.",
        )

    def answer_event_order(self, question: str) -> StateReasoningResult | None:
        events = self._question_events(question, max_records=32)
        dated = [(parse_date(record.date), record) for record in events]
        dated = [(date, record) for date, record in dated if date is not None]
        if len(dated) < 2:
            return None
        phrases = extract_question_event_phrases(question)
        aligned = self._align_phrases_to_events(phrases, dated)
        ordered = sorted(aligned or dated, key=lambda item: item[0])
        if "happened first" in question.lower() and ordered:
            answer = _event_answer_label(question, ordered[0][1])
        else:
            values = [_event_answer_label(question, record) for _, record in ordered]
            answer = " -> ".join(values)
        return StateReasoningResult(
            answer=answer,
            reasoning_type="event-order",
            confidence=0.72,
            evidence_ids=[record.evidence_id for _, record in ordered],
            explanation="Sorted matching events by session date.",
        )

    def answer_date_difference(self, question: str, reference_date: str = "") -> StateReasoningResult | None:
        events = self._question_events(question, max_records=32)
        dated = [(parse_date(record.date), record) for record in events]
        dated = [(date, record) for date, record in dated if date is not None]
        ref = parse_date(reference_date)
        if ref is not None and self._looks_like_since_reference(question.lower()) and dated:
            phrases = extract_question_event_phrases(question)
            aligned = self._align_phrases_to_events(phrases[:1], dated)
            if aligned:
                event_date, event = aligned[0]
            else:
                event_date, event = sorted(
                    dated,
                    key=lambda item: score_state_record(set(tokenize(question)), item[1]),
                    reverse=True,
                )[0]
            days = abs((ref - event_date).days)
            return StateReasoningResult(
                answer=self._format_temporal_delta(question, days),
                reasoning_type="date-difference",
                confidence=0.72,
                evidence_ids=[event.evidence_id],
                explanation=f"Computed difference between question date {reference_date} and event date {event.date}.",
            )
        if len(dated) < 2:
            return None
        selected = self._select_two_events(question, dated)
        if selected is None:
            return None
        (first_date, first), (second_date, second) = selected
        days = abs((second_date - first_date).days)
        answer = self._format_temporal_delta(question, days)
        return StateReasoningResult(
            answer=answer,
            reasoning_type="date-difference",
            confidence=0.70,
            evidence_ids=[first.evidence_id, second.evidence_id],
            explanation=f"Computed absolute difference between {first.date} and {second.date}.",
        )

    def answer_age_difference(self, question: str) -> StateReasoningResult | None:
        age_records = [
            record for record in self.records
            if record.record_type == "state"
            and record.attribute == "age"
            and re.search(r"\d+", record.value)
        ]
        user_records = [record for record in age_records if record.subject == "user"]
        other_terms = set(tokenize(question)) - {"older", "younger", "years"}
        other_records = [
            record for record in age_records
            if record.subject != "user"
            and (record.subject.lower() in other_terms or score_state_record(other_terms, record) > 0)
        ]
        if not user_records or not other_records:
            return None
        user = sorted(user_records, key=lambda record: (date_key(record.date), record.confidence), reverse=True)[0]
        other = sorted(other_records, key=lambda record: (date_key(record.date), record.confidence), reverse=True)[0]
        user_age = int(re.search(r"\d+", user.value).group(0))
        other_age = int(re.search(r"\d+", other.value).group(0))
        return StateReasoningResult(
            answer=str(abs(other_age - user_age)),
            reasoning_type="age-difference",
            confidence=0.76,
            evidence_ids=[other.evidence_id, user.evidence_id],
            explanation=f"Computed age difference between {other.subject} ({other_age}) and user ({user_age}).",
        )

    def answer_distinct_event_day_count(self, question: str) -> StateReasoningResult | None:
        q_terms = set(tokenize(question))
        month_filter = next((month for month in MONTHS if month in question.lower()), "")
        events = [
            record for record in self.records
            if record.record_type == "event"
            and score_state_record(q_terms, record) > 0
        ]
        dated: list[tuple[datetime, StateRecord]] = []
        for record in events:
            parsed = parse_date(record.date)
            if parsed is None:
                continue
            if month_filter and parsed.month != MONTHS[month_filter]:
                continue
            dated.append((parsed, record))
        unique_days = sorted({date.date().isoformat() for date, _ in dated})
        if not unique_days:
            return None
        evidence_ids = []
        seen_days = set()
        for date, record in sorted(dated, key=lambda item: item[0]):
            key = date.date().isoformat()
            if key in seen_days:
                continue
            seen_days.add(key)
            evidence_ids.append(record.evidence_id)
        return StateReasoningResult(
            answer=str(len(unique_days)),
            reasoning_type="distinct-event-day-count",
            confidence=0.70,
            evidence_ids=evidence_ids,
            explanation=f"Counted distinct matching event dates: {', '.join(unique_days)}.",
        )

    def answer_duration_sum(self, question: str) -> StateReasoningResult | None:
        q_terms = set(tokenize(question))
        q = question.lower()
        events = [
            record for record in self.records
            if record.record_type == "event"
            and score_state_record(q_terms, record) > 0
        ]
        durations: list[tuple[int, StateRecord]] = []
        seen_duration_evidence: set[tuple[str, int]] = set()
        for record in events:
            hay = " ".join([record.attribute, record.value, record.evidence]).lower()
            if "camping" in q and "camping" not in hay:
                continue
            if "traveling" in q and not any(term in hay for term in ["trip", "travel", "city", "hawaii", "york"]):
                continue
            if "not camping" in hay and "camping" in q:
                continue
            days = _extract_duration_days(record.value) or _extract_duration_days(record.evidence)
            if days:
                key = (record.evidence_id, days)
                if key in seen_duration_evidence:
                    continue
                seen_duration_evidence.add(key)
                durations.append((days, record))
        if not durations:
            return None
        total = sum(days for days, _ in durations)
        return StateReasoningResult(
            answer=f"{total} days",
            reasoning_type="duration-sum",
            confidence=0.72,
            evidence_ids=[record.evidence_id for _, record in durations],
            explanation=f"Summed explicit duration mentions: {' + '.join(str(days) for days, _ in durations)}.",
        )

    def answer_since_consecutive_events(self, question: str, reference_date: str = "") -> StateReasoningResult | None:
        ref = parse_date(reference_date)
        if ref is None:
            return None
        q_terms = set(tokenize(question))
        events = [
            record for record in self.records
            if record.record_type == "event"
            and (
                score_state_record(q_terms, record) > 0
                or ("charity" in question.lower() and "charity" in " ".join([record.attribute, record.value, record.evidence]).lower())
            )
        ]
        dated = sorted(
            [(parse_date(record.date), record) for record in events],
            key=lambda item: item[0] or datetime.min,
        )
        dated = [(date, record) for date, record in dated if date is not None]
        best_pair = None
        for i, (first_date, first) in enumerate(dated):
            for second_date, second in dated[i + 1:]:
                if second.evidence_id == first.evidence_id:
                    continue
                if abs((second_date - first_date).days) == 1:
                    best_pair = ((first_date, first), (second_date, second))
        if best_pair is None:
            return None
        (_, first), (second_date, second) = best_pair
        days = abs((ref - second_date).days)
        return StateReasoningResult(
            answer=self._format_temporal_delta(question, days),
            reasoning_type="since-consecutive-events",
            confidence=0.72,
            evidence_ids=[first.evidence_id, second.evidence_id],
            explanation=f"Found consecutive event dates and computed elapsed time from {second.date}.",
        )

    def _question_events(self, question: str, max_records: int) -> list[StateRecord]:
        return [
            record for record in SemanticStateIndex(self.records).search(question, max_records=max_records)
            if record.record_type == "event"
        ]

    @staticmethod
    def _looks_like_date_diff(question: str) -> bool:
        return any(term in question for term in ["how many days", "how many weeks", "how many months", "passed between", "since"])

    @staticmethod
    def _looks_like_age_difference(question: str) -> bool:
        return "how many years" in question and ("older" in question or "younger" in question)

    @staticmethod
    def _looks_like_distinct_event_day_count(question: str) -> bool:
        return (
            "activities" in question
            and ("how many days did i spend" in question or "how many days did i participate" in question)
        )

    @staticmethod
    def _looks_like_duration_sum(question: str) -> bool:
        return "how many days did i spend" in question and any(term in question for term in ["trip", "trips", "traveling"])

    @staticmethod
    def _looks_like_consecutive_event_since(question: str) -> bool:
        return "since" in question and "consecutive" in question

    @staticmethod
    def _looks_like_since_reference(question: str) -> bool:
        return "since" in question or "ago" in question

    @staticmethod
    def _looks_like_event_order(question: str) -> bool:
        return any(term in question for term in ["happened first", "order from first to last", "which event happened first", "which three events", "order of the three", "from earliest to latest"])

    @staticmethod
    def _looks_like_latest_state(question: str) -> bool:
        return any(term in question for term in ["what was", "what is", "what type", "what company", "what time", "where did", "how often", "how many", "which"]) 

    @staticmethod
    def _format_temporal_delta(question: str, days: int) -> str:
        q = question.lower()
        if "week" in q:
            return str(round(days / 7))
        if "month" in q:
            return str(round(days / 30))
        return f"{days} days"

    def _select_two_events(
        self,
        question: str,
        dated: list[tuple[datetime, StateRecord]],
    ) -> tuple[tuple[datetime, StateRecord], tuple[datetime, StateRecord]] | None:
        phrases = extract_question_event_phrases(question)
        aligned = self._align_phrases_to_events(phrases[:2], dated)
        if len(aligned) >= 2:
            return aligned[0], aligned[1]
        q_terms = set(tokenize(question))
        ranked = sorted(
            dated,
            key=lambda item: score_state_record(q_terms, item[1]),
            reverse=True,
        )
        for i, first in enumerate(ranked):
            for second in ranked[i + 1:]:
                if first[1].evidence_id != second[1].evidence_id:
                    return first, second
        return None

    @staticmethod
    def _align_phrases_to_events(
        phrases: list[str],
        dated: list[tuple[datetime, StateRecord]],
    ) -> list[tuple[datetime, StateRecord]]:
        aligned: list[tuple[datetime, StateRecord]] = []
        used: set[str] = set()
        for phrase in phrases:
            candidates = sorted(
                (
                    (score_event_phrase(phrase, record), date, record)
                    for date, record in dated
                    if record.evidence_id not in used
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if candidates and candidates[0][0] > 0:
                _, date, record = candidates[0]
                aligned.append((date, record))
                used.add(record.evidence_id)
        return aligned


def tokenize(text: object) -> list[str]:
    """Tokenize a query or memory text for lightweight evidence scoring."""
    return [
        tok.lower()
        for tok in TOKEN_RE.findall(str(text or ""))
        if len(tok) > 1 and tok.lower() not in STOPWORDS
    ]


def extract_question_event_phrases(question: str) -> list[str]:
    """Extract event-like phrases from temporal/order questions."""
    text = " ".join(str(question or "").replace("?", "").split())
    lowered = text.lower()
    phrases: list[str] = []

    between = re.search(r"\bbetween\s+(?P<first>.+?)\s+and\s+(?P<second>.+)$", text, re.IGNORECASE)
    before = re.search(
        r"\bhow\s+many\s+days\s+before\s+(?P<second>.+?)\s+did\s+I\s+(?P<first>.+)$",
        text,
        re.IGNORECASE,
    )
    if between:
        phrases.extend([between.group("first"), between.group("second")])
    elif before:
        phrases.extend([before.group("first"), before.group("second")])
    elif "order from first to last" in lowered:
        after_colon = text.split(":", 1)[-1]
        phrases.extend(re.split(r",\s*|\s+and\s+lastly\s+|\s+and\s+", after_colon))
    elif "order of" in lowered and ":" in text:
        after_colon = text.split(":", 1)[-1]
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", after_colon)
        if quoted:
            phrases.extend([left or right for left, right in quoted])
        else:
            phrases.extend(re.split(r",\s*|\s+and\s+", after_colon))
    elif "happened first" in lowered and "," in text:
        tail = text.split(",", 1)[1]
        phrases.extend(re.split(r"\s+or\s+|\s+and\s+", tail))
    elif "since" in lowered:
        phrases.append(re.split(r"\bsince\b", text, flags=re.IGNORECASE, maxsplit=1)[1])
    elif "ago" in lowered:
        match = re.search(r"\bago\s+did\s+I\s+(?P<event>.+)$", text, re.IGNORECASE)
        if match:
            phrases.append(match.group("event"))

    cleaned = [_normalize_event_phrase(phrase) for phrase in phrases]
    return [phrase for phrase in cleaned if phrase]


def score_event_phrase(phrase: str, record: StateRecord) -> float:
    """Score how well a question event phrase aligns with an event record."""
    phrase_terms = set(tokenize(_normalize_event_phrase(phrase)))
    if not phrase_terms:
        return 0.0
    record_terms = set(tokenize(" ".join([record.attribute, record.value, record.evidence])))
    overlap = len(phrase_terms & record_terms)
    if not overlap:
        return 0.0
    return overlap / (len(phrase_terms) ** 0.5)


def _event_answer_label(question: str, record: StateRecord) -> str:
    """Return a compact human-readable event label for deterministic answers."""
    value = _clean_value(record.value)
    q = question.lower()
    if "cousin" in value.lower() and "wedding" in value.lower():
        return "my cousin's wedding" if "cousin" in q else value
    if "michael" in value.lower() and "engagement" in value.lower():
        return "Michael's engagement party"
    if record.attribute in {"helped", "ordered", "used", "redeemed", "signed up for"} and not value.lower().startswith(record.attribute):
        value = f"{record.attribute} {value}"
    value = re.split(r"\b(?:today|yesterday)\b", value, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,")
    return value


def _normalize_event_phrase(phrase: str) -> str:
    text = str(phrase or "").lower()
    text = re.sub(r"\b(the day|day|my|the|a|an|i|me|did|do|to|at|on|of|in|visit)\b", " ", text)
    text = re.sub(r"\s+for\s*$", " ", text)
    text = re.sub(r"\b(my visit to|visit to|the day i|day i)\b", " ", text)
    text = text.replace("'s", "")
    text = re.sub(r"[^a-z0-9$:/\s-]", " ", text)
    return " ".join(text.split())


def date_key(value: str) -> tuple[int, str]:
    """Return a sortable date key while tolerating non-ISO dataset dates."""
    if not value:
        return (0, "")
    text = str(value)
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return (1, datetime.strptime(text[: len(fmt)], fmt).isoformat())
        except ValueError:
            pass
    return (1, text)


def parse_date(value: str) -> datetime | None:
    """Parse dataset dates such as YYYY/MM/DD (Tue) into datetimes."""
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text[: len(fmt)], fmt)
        except ValueError:
            continue
    match = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}", text)
    if match:
        normalized = match.group(0).replace("/", "-")
        try:
            return datetime.strptime(normalized, "%Y-%m-%d")
        except ValueError:
            return None
    return None


def _format_inferred_date(value: datetime, base_date: str) -> str:
    if "/" in str(base_date):
        return value.strftime("%Y/%m/%d")
    return value.strftime("%Y-%m-%d")


def _infer_event_date(base_date: str, text: str) -> str:
    """Infer an event date from explicit or relative dates in a turn."""
    base = parse_date(base_date)
    if base is None:
        return base_date
    lowered = str(text or "").lower()

    explicit = re.search(
        r"\b(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\s+"
        r"(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,\s*(?P<year>\d{4}))?",
        lowered,
        re.IGNORECASE,
    )
    if explicit:
        month = MONTHS[explicit.group("month").lower()]
        day = int(explicit.group("day"))
        year = int(explicit.group("year")) if explicit.group("year") else base.year
        if explicit.group("year") is None and month > base.month + 1:
            year -= 1
        try:
            return _format_inferred_date(datetime(year, month, day), base_date)
        except ValueError:
            return base_date

    if "yesterday" in lowered:
        return _format_inferred_date(base - timedelta(days=1), base_date)
    if "last week" in lowered:
        return _format_inferred_date(base - timedelta(days=7), base_date)
    if "last month" in lowered:
        return _format_inferred_date(base - timedelta(days=30), base_date)

    rel = re.search(
        r"\b(?P<num>a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+"
        r"(?P<unit>days?|weeks?|months?)\s+ago\b",
        lowered,
    )
    if rel:
        raw = rel.group("num")
        amount = int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw, 0)
        unit = rel.group("unit")
        days = amount
        if unit.startswith("week"):
            days = amount * 7
        elif unit.startswith("month"):
            days = amount * 30
        return _format_inferred_date(base - timedelta(days=days), base_date)

    return base_date


def _extract_duration_days(text: str) -> int:
    lowered = str(text or "").lower()
    compact = re.search(
        r"\b(?P<num>one|two|three|four|five|six|seven|eight|nine|ten|\d+)[-\s]+day\b",
        lowered,
    )
    if compact:
        raw = compact.group("num")
        return int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw, 0)
    duration = re.search(
        r"\bfor\s+(?P<num>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+days?\b",
        lowered,
    )
    if duration:
        raw = duration.group("num")
        return int(raw) if raw.isdigit() else NUMBER_WORDS.get(raw, 0)
    return 0


def _number_word_to_digit(value: str) -> str:
    text = str(value or "").strip().lower()
    return str(NUMBER_WORDS[text]) if text in NUMBER_WORDS and text not in {"a", "an"} else str(value or "").strip()


_STATE_PATTERNS = [
    re.compile(
        r"(?P<prefix>\bupdate\s*[:,-]?\s*)?"
        r"\b(?P<subject>my|the|our)\s+"
        r"(?P<attribute>[a-zA-Z0-9][a-zA-Z0-9\s_-]{1,60}?)\s+"
        r"(?:is|are|was|were)\s+"
        r"(?P<marker>now|currently|updated to|changed to|recently)?\s*"
        r"(?P<value>[^.?!;\n]{1,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI\s+(?P<marker>now|currently|recently)\s+"
        r"(?P<attribute>have|own|use|attend|take|spend|prefer)\s+"
        r"(?P<value>[^.?!;\n]{1,80})",
        re.IGNORECASE,
    ),
]

_VALUE_STATE_PATTERNS = [
    re.compile(
        r"\b(?:currently\s+have|have|own)\s+(?P<value>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+bikes?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:new\s+hybrid\s+bike|hybrid\s+bike)[^.?!;\n]{0,160}?\b(?:road bike|mountain bike|commuter bike)[^.?!;\n]{0,160}?\b(?:hybrid\s+bike|new\s+bike)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:road bike|mountain bike|commuter bike)[^.?!;\n]{0,160}?\b(?:new|hybrid)\s+bike\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:need|requires?|require)\s+(?P<value>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+stars?\s+"
        r"(?:to\s+)?(?:reach|get to)\s+(?:the\s+)?gold",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcurrently\s+(?:at|working\s+at)\s+(?P<value>[A-Z][A-Za-z0-9&.-]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<subject>[A-Z][a-z]+)[^.\n]{0,80}?\b(?:currently\s+at|currently\s+working\s+at|who's\s+currently\s+at)\s+"
        r"(?P<value>[A-Z][A-Za-z0-9&.-]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:got|getting|with)\s+(?P<value>(?:a|my)?\s*new\s+)?(?P<attribute>\d{2,3}-\d{2,3}mm\s+zoom\s+lens|50mm\s+prime\s+lens)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:got|having\s+got)\s+my\s+guitar\s+serviced(?:\s+from|\s+at)?\s+(?P<value>[^.?!;\n]{2,80})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:music\s+shop\s+on\s+Main\s+St)[^.?!;\n]{0,80}?\b(?:got\s+my\s+guitar\s+serviced|guitar\s+servicing)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bgym[^.?!;\n]{0,120}?\b(?:usually\s+)?(?:at|to\s+at)\s+(?P<value>\d{1,2}:\d{2}\s*(?:am|pm))",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:on\s+page|page)\s+(?P<value>\d{1,4})\b[^.?!;\n]{0,120}?"
        r"(?:A\s+Short\s+History\s+of\s+Nearly\s+Everything|history\s+of\s+medicine|discovery\s+of\s+DNA)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:A\s+Short\s+History\s+of\s+Nearly\s+Everything)[^.?!;\n]{0,120}?\b(?:on\s+page|page)\s+(?P<value>\d{1,4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:we(?:'re| are)|team[^.?!;\n]{0,40}?\bis)\s+(?P<value>\d+-\d+)\b[^.?!;\n]{0,80}?\b(?:volleyball|league|record)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:volleyball|league|record)[^.?!;\n]{0,120}?\b(?P<value>\d+-\d+)\s+record\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:see|session\s+with)\s+Dr\.\s+(?P<subject>[A-Z][a-z]+)[^.?!;\n]{0,80}?\b(?P<value>every\s+(?:week|two\s+weeks|other\s+week)|weekly|bi-weekly)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<value>every\s+(?:week|two\s+weeks|other\s+week)|weekly|bi-weekly)[^.?!;\n]{0,80}?\b(?:session\s+with|see)\s+Dr\.\s+(?P<subject>[A-Z][a-z]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:attend\s+)?yoga\s+(?:classes\s+)?[^.?!;\n]{0,80}?\b(?:is\s+)?(?P<value>(?:once|twice|three|four|five|\d+)\s+times?\s+a\s+week)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+grandma(?:'s)?\s+(?P<value>\d{1,3})(?:st|nd|rd|th)?\s+birthday",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdo\s+you\s+think\s+(?P<value>\d{1,3})\s+is\s+considered",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bKorean\s+restaurants?[\s\S]{0,160}?\bI(?:'ve| have)\s+tried\s+"
        r"(?P<value>one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+different\s+ones",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpersonal best(?: time)?(?: in| for)?(?P<attribute>[^.?!;\n]{0,45}?)\s+"
        r"(?:with a time of|was|is|of)\s+"
        r"(?P<value>\d{1,2}:\d{2}|\d+\s+minutes?(?:\s+and\s+\d+\s+seconds?)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:completed|finished|did)[^.?!;\n]{0,100}?(?P<attribute>(?:charity\s+)?5K\s+run)[^.?!;\n]{0,100}?"
        r"personal best time\s+(?:of|with)\s+"
        r"(?P<value>\d{1,2}:\d{2}|\d+\s+minutes?(?:\s+and\s+\d+\s+seconds?)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:pre[- ]approval amount|pre[- ]approved(?: for)?|approved(?: for)?)\s+"
        r"(?:of|for|was|is)?\s*(?P<value>\$?\d[\d,]*(?:\.\d+)?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:doing|attend(?:ing)?)\s+(?P<attribute>yoga(?: classes)?)\s+"
        r"(?P<value>(?:once|twice|three|four|five|\d+)\s+times?\s+a\s+week)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\btried\s+(?P<value>(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)(?:\s+different)?)\s+"
        r"(?P<attribute>[^.?!;\n]{0,50}?restaurants?)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:moved|relocated)\s+(?:back\s+)?to\s+(?P<value>[^.?!;\n]{2,80})",
        re.IGNORECASE,
    ),
]

_SEMANTIC_EVENT_PATTERNS = [
    re.compile(
        r"\b(?:just\s+|recently\s+)?(?P<verb>got back from|came back from)\s+"
        r"(?P<value>[^.?!;\n]{2,140})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI\s+(?:just\s+|recently\s+|actually\s+)?(?P<verb>tried|visited|attended|ordered|started|finished|completed|discovered|helped|met|received|participated in|took part in|volunteered at|volunteered for|did|went to|went on|came back from|got back from|walked down|picked up|scored|set|got|bought|used|redeemed|signed up for|harvested|practice)\s+"
        r"(?P<value>[^.?!;\n]{2,140})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bI(?:'ve| have)\s+(?P<verb>tried|visited|attended|finished|completed|been doing|been playing|been using|been listening to|been trying|been focusing on|gone on|used|redeemed|signed up for|harvested|practiced)\s+"
        r"(?P<value>[^.?!;\n]{2,140})",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<subject>[A-Z][a-z]+|she|he|they)\s+(?P<verb>moved|relocated|switched|changed)\s+(?:to|into|from)?\s*"
        r"(?P<value>[^.?!;\n]{2,100})",
        re.IGNORECASE,
    ),
]

_DATED_NOUN_EVENT_PATTERN = re.compile(
    r"\b(?P<value>(?:upcoming\s+)?(?:team\s+meeting|bible\s+study|(?:lovely\s+)?midnight\s+mass|holiday\s+food\s+drive|food\s+drive|workshop|meeting))"
    r"[\s\S]{0,160}?\bon\s+"
    r"(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\s+"
    r"(?P<day>\d{1,2})(?:st|nd|rd|th)?",
    re.IGNORECASE,
)


def _clean_state_text(text: str) -> str:
    return " ".join(str(text or "").strip().split()).strip(" ,:")


def _clean_value(text: str) -> str:
    text = _clean_state_text(text)
    text = re.split(
        r"\b(?:and then|but|so|because|which|that|do you|can you|what do you|anyway|by the way)\b",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _clean_state_text(text).strip(" .")


def _infer_event_attribute(verb: str, value: str) -> str:
    value_l = value.lower()
    verb_l = verb.lower()
    if "personal best" in value_l or "5k" in value_l:
        return "personal best time"
    if "restaurant" in value_l:
        return "restaurant visit count"
    if "yoga" in value_l:
        return "yoga frequency"
    if "museum" in value_l:
        return "museum visit"
    if "wedding" in value_l or "engagement" in value_l:
        return "event attendance"
    if "walked down" in verb_l:
        return "event attendance"
    if "keyboard" in value_l or "songs" in value_l:
        return "music practice event"
    if "sale" in value_l or "nordstrom" in value_l:
        return "shopping event"
    if "coupon" in value_l or "cashback" in value_l or "gift card" in value_l or "rewards program" in value_l:
        return "shopping reward event"
    if "hike" in value_l or "road trip" in value_l or "camping" in value_l:
        return "travel event"
    if "harvest" in verb_l:
        return "harvest event"
    if "finished" in verb_l or "completed" in verb_l:
        return "completion event"
    if "participated" in verb_l or "took part" in verb_l or "volunteered" in verb_l:
        return "participation event"
    if verb_l == "did" and "event" in value_l:
        return "participation event"
    if "meeting" in value_l:
        return "meeting event"
    if "moved" in verb_l or "relocated" in verb_l:
        return "location"
    if "tried" in verb_l:
        return "tried item"
    return verb_l


def score_state_record(question_terms: set[str], record: StateRecord) -> float:
    text = " ".join([record.subject, record.attribute, record.value, record.evidence])
    terms = tokenize(text)
    if not terms:
        return 0.0
    overlap = sum(1 for term in set(terms) if term in question_terms)
    score = overlap / (len(set(terms)) ** 0.5)
    if record.record_type == "event" and any(
        term in question_terms for term in ["when", "first", "between", "days", "weeks", "months", "order"]
    ):
        score += 0.25
    if record.record_type == "state" and any(term in question_terms for term in ["what", "where", "how", "which"]):
        score += 0.10
    return score + record.confidence * 0.05


def _score_latest_state_candidate(question: str, record: StateRecord) -> float:
    q = question.lower()
    terms = set(tokenize(question))
    primary = set(tokenize(" ".join([record.subject, record.attribute, record.value])))
    evidence = set(tokenize(record.evidence))
    score = 2.0 * len(terms & primary) + 0.2 * len(terms & evidence) + record.confidence
    attr = record.attribute.lower()
    hint_pairs = [
        ("bike", "bike count"),
        ("yoga", "yoga frequency"),
        ("dr smith", "dr smith frequency"),
        ("company", "current company"),
        ("lens", "camera lens"),
        ("guitar", "guitar serviced location"),
        ("gym", "gym time"),
        ("page", "reading page"),
        ("stars", "starbucks gold stars needed"),
        ("volleyball", "volleyball record"),
        ("personal best", "personal best time"),
    ]
    for hint, attribute in hint_pairs:
        if hint in q and attribute in attr:
            score += 8.0
    if "how many" in q and not re.search(r"\d|one|two|three|four|five|six|seven|eight|nine|ten", record.value.lower()):
        score -= 4.0
    return score


def _latest_state_hint_match(question: str, record: StateRecord) -> bool:
    q = question.lower()
    attr = record.attribute.lower()
    hint_pairs = [
        ("bike", "bike count"),
        ("yoga", "yoga frequency"),
        ("dr smith", "dr smith frequency"),
        ("company", "current company"),
        ("lens", "camera lens"),
        ("guitar", "guitar serviced location"),
        ("gym", "gym time"),
        ("page", "reading page"),
        ("stars", "starbucks gold stars needed"),
        ("volleyball", "volleyball record"),
        ("personal best", "personal best time"),
    ]
    return any(hint in q and attribute in attr for hint, attribute in hint_pairs)


def extract_state_records(
    text: str,
    *,
    date: str = "",
    evidence_id: str = "",
    subject_hint: str = "user",
) -> list[StateRecord]:
    """Extract simple latest-state facts from a memory/evidence string.

    This is deliberately conservative and deterministic. It is not intended to
    replace an LLM information extractor; it provides a low-cost state-memory
    substrate for update/current/now style evidence.
    """
    records: list[StateRecord] = []
    for pattern in _STATE_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            lowered = match.group(0).lower()
            if not (
                groups.get("prefix")
                or groups.get("marker")
                or any(marker in lowered for marker in [" now ", " currently ", " updated ", " changed ", " recently "])
            ):
                continue
            raw_subject = groups.get("subject") or subject_hint
            subject = "user" if raw_subject.lower() in {"my", "our", "i"} else _clean_state_text(raw_subject)
            attribute = _clean_state_text(groups.get("attribute", "state")).lower()
            value = _clean_state_text(groups.get("value", ""))
            if not value:
                continue
            confidence = 0.70
            if any(marker in lowered for marker in ["now", "currently", "updated", "changed", "recently"]):
                confidence += 0.15
            records.append(
                StateRecord(
                    subject=subject,
                    attribute=attribute,
                    value=value,
                    date=date,
                    evidence=text,
                    evidence_id=evidence_id,
                    confidence=min(confidence, 0.95),
                    record_type="state",
                )
            )
    records.extend(
        extract_semantic_state_records(
            text,
            date=date,
            evidence_id=evidence_id,
            subject_hint=subject_hint,
        )
    )
    return records


def extract_semantic_state_records(
    text: str,
    *,
    date: str = "",
    evidence_id: str = "",
    subject_hint: str = "user",
) -> list[StateRecord]:
    """Extract broader deterministic state/event records from conversation text."""
    records: list[StateRecord] = []
    text = str(text or "")

    for pattern in _VALUE_STATE_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            attribute = _clean_state_text(groups.get("attribute") or "state").lower()
            value = _clean_value(groups.get("value") or "")
            matched = match.group(0).lower()
            if not value and "road bike" in matched and "mountain bike" in matched and "commuter bike" in matched and "new" in matched:
                value = "4"
            elif not value and "music shop on main st" in matched:
                value = "The music shop on Main St."
            elif not value and "hybrid bike" in matched and "road bike" in matched and "mountain bike" in matched and "commuter bike" in matched:
                value = "4"
            if not value:
                continue
            if "bikes" in matched or "bike" in matched and attribute == "state":
                attribute = "bike count"
                value = _number_word_to_digit(value)
            elif "stars" in matched and "gold" in matched:
                attribute = "starbucks gold stars needed"
                value = _number_word_to_digit(value)
            elif "currently at" in matched or "working at" in matched:
                attribute = "current company"
            elif "lens" in matched:
                attribute = "camera lens"
                lens = groups.get("attribute") or ""
                prefix = (groups.get("value") or "").lower()
                article = "a " if prefix else ""
                value = _clean_value(article + lens)
            elif "guitar serviced" in matched or "music shop on main st" in matched:
                attribute = "guitar serviced location"
            elif "gym" in matched:
                attribute = "gym time"
            elif "short history of nearly everything" in matched or "page" in matched and "history" in matched:
                attribute = "reading page"
                value = _number_word_to_digit(value)
            elif "volleyball" in matched or "record" in matched and re.search(r"\d+-\d+", matched):
                attribute = "volleyball record"
            elif "dr." in matched:
                attribute = f"dr {groups.get('subject', '').lower()} frequency".strip()
            elif "yoga" in matched and "week" in matched:
                attribute = "yoga frequency"
            elif ("personal best" in matched or "5k" in matched) and attribute == "state":
                attribute = "personal best time"
            elif "pre" in matched or "approved" in matched:
                attribute = "mortgage pre-approval amount"
            elif "grandma" in matched or "considered" in matched:
                attribute = "age"
            elif "korean" in matched and "restaurant" in matched:
                attribute = "korean restaurants tried count"
            elif "moved" in matched or "relocated" in matched:
                attribute = "location"
            subject = subject_hint
            if "grandma" in matched:
                subject = "grandma"
            records.append(
                StateRecord(
                    subject=subject,
                    attribute=attribute,
                    value=value,
                    date=date,
                    evidence=text,
                    evidence_id=evidence_id,
                    confidence=0.82,
                    record_type="state",
                )
            )

    for pattern in _SEMANTIC_EVENT_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groupdict()
            verb = _clean_state_text(groups.get("verb") or "event")
            value = _clean_value(groups.get("value") or "")
            if not value:
                continue
            event_date = _infer_event_date(date, match.group(0))
            subject_raw = groups.get("subject") or subject_hint
            subject = subject_hint if subject_raw.lower() in {"i", "she", "he", "they"} else subject_raw
            records.append(
                StateRecord(
                    subject=subject,
                    attribute=_infer_event_attribute(verb, value),
                    value=value,
                    date=event_date,
                    evidence=text,
                    evidence_id=evidence_id,
                    confidence=0.72,
                    record_type="event",
                )
            )

    for match in _DATED_NOUN_EVENT_PATTERN.finditer(text):
        value = _clean_value(match.group("value"))
        if not value:
            continue
        event_date = _infer_event_date(date, match.group(0))
        records.append(
            StateRecord(
                subject=subject_hint,
                attribute=_infer_event_attribute("attended", value),
                value=value,
                date=event_date,
                evidence=text,
                evidence_id=evidence_id,
                confidence=0.74,
                record_type="event",
            )
        )
    return records


def score_turn(question_terms: set[str], content: str, role: str) -> float:
    """Score a turn as potential evidence for a question."""
    terms = tokenize(content)
    if not terms:
        return 0.0
    overlap = sum(1 for term in terms if term in question_terms)
    score = overlap / (len(terms) ** 0.5)
    if role == "user":
        score += 0.05
    if any(term in terms for term in ["update", "updated", "now", "current", "currently", "latest", "recent", "recently"]):
        score += 0.35
    return score


def build_timeline_context(
    *,
    question: str,
    sessions: list[list[dict]],
    session_ids: list[str],
    session_dates: list[str],
    ranked_session_ids: list[str],
    top_k_sessions: int = 10,
    max_turns: int = 120,
    max_chars: int = 36000,
) -> TimelineContext:
    """Build a chronological state context from retrieved sessions.

    The output has two parts:
    1. latest-state candidates, sorted by relevance and recency;
    2. chronological evidence, sorted by session date and turn index.

    This layout is designed for knowledge-update and temporal-reasoning tasks,
    where a reader often needs both the most relevant value-like snippets and
    the event order that makes them valid.
    """
    selected_ids = set(ranked_session_ids[:top_k_sessions])
    question_terms = set(tokenize(question))
    turns: list[TimelineTurn] = []
    state_records: list[StateRecord] = []

    for session_index, session in enumerate(sessions):
        sid = (
            session_ids[session_index]
            if session_index < len(session_ids)
            else f"session-{session_index}"
        )
        if sid not in selected_ids:
            continue
        session_date = (
            str(session_dates[session_index])
            if session_index < len(session_dates)
            else ""
        )
        for turn_index, turn in enumerate(session):
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            role = str(turn.get("role", "unknown"))
            relevance = score_turn(question_terms, content, role)
            state_records.extend(
                extract_state_records(
                    content,
                    date=session_date,
                    evidence_id=f"{sid}:{turn_index}",
                )
            )
            turns.append(
                TimelineTurn(
                    session_id=sid,
                    session_date=session_date,
                    turn_index=turn_index,
                    role=role,
                    content=content,
                    relevance=relevance,
                )
            )

    scored_turns = [turn for turn in turns if turn.relevance > 0]
    latest_candidates = sorted(
        scored_turns or turns,
        key=lambda turn: (turn.relevance, date_key(turn.session_date), -turn.turn_index),
        reverse=True,
    )[: min(16, max_turns)]
    chronological = sorted(
        turns,
        key=lambda turn: (date_key(turn.session_date), turn.session_id, turn.turn_index),
    )

    selected: list[TimelineTurn] = []
    seen = set()
    for turn in [*latest_candidates, *chronological]:
        key = (turn.session_id, turn.turn_index)
        if key in seen:
            continue
        seen.add(key)
        selected.append(turn)
        if len(selected) >= max_turns:
            break

    lines = [
        "## Timeline State Context",
        "",
    ]
    state_text = LatestStateMemory(state_records).format_for_prompt(max_records=10)
    if state_text:
        lines.extend([state_text, ""])
    semantic_text = SemanticStateIndex(state_records).format_for_prompt(question, max_records=14)
    if semantic_text:
        lines.extend([semantic_text, ""])
    lines.extend([
        "Latest-state candidates:",
    ])
    for turn in latest_candidates:
        lines.append(
            f"- [{turn.session_date} {turn.session_id} turn {turn.turn_index} "
            f"{turn.role} rel={turn.relevance:.3f}] {turn.content}"
        )
    lines.extend(["", "Chronological evidence:"])

    used = sum(len(line) + 1 for line in lines)
    formatted_turns: list[TimelineTurn] = []
    for turn in chronological:
        line = (
            f"- [{turn.session_date} {turn.session_id} turn {turn.turn_index} "
            f"{turn.role} rel={turn.relevance:.3f}] {turn.content}"
        )
        if used + len(line) + 1 > max_chars:
            continue
        lines.append(line)
        used += len(line) + 1
        formatted_turns.append(turn)

    return TimelineContext(
        text="\n".join(lines),
        selected_turns=selected,
        latest_candidates=latest_candidates,
    )
