"""Date normalization for tsvector injection.

At ingest time, date expressions in memory content are detected and
expanded into multiple normalized forms (ISO 8601, US, UK, spelled-out).
These are injected into the tsvector at weight 'B' so that full-text
search queries for any date format can match the stored memory.

Problem this solves: The stored turn says "May 1st, 2022" but the
benchmark query says "1 May, 2022" or "2022-05-01". Without
normalization the tsvector index sees them as unrelated tokens and
FTS fails before ranking even begins.

This is pure CPU-side string processing — no DB round-trips.
"""

from __future__ import annotations

import re

# ── Month name lookup ──────────────────────────────────────────────────────

_MONTH_NAMES = [
    "", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
_MONTH_ABBR = [
    "", "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]
_MONTH_NAME_TO_NUM: dict[str, int] = {}
for _i, (_name, _abbr) in enumerate(zip(_MONTH_NAMES[1:], _MONTH_ABBR[1:]), start=1):
    _MONTH_NAME_TO_NUM[_name] = _i
    _MONTH_NAME_TO_NUM[_abbr] = _i

# ── Regex patterns ─────────────────────────────────────────────────────────

# Matches: "May 1, 2022" / "May 1st, 2022" / "1 May 2022" / "1st May 2022"
_NAMED_DATE_RE = re.compile(
    r"\b(?:"
    r"(?P<mn1>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(?P<d1>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y1>\d{4})"
    r"|"
    r"(?P<d2>\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(?P<mn2>jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+(?P<y2>\d{4})"
    r")\b",
    re.IGNORECASE,
)

# Matches: "2022-05-01" / "2022/05/01"
_ISO_DATE_RE = re.compile(
    r"\b(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})\b"
)

# Matches: "05/01/2022" or "01/05/2022" (ambiguous — emit both interpretations)
_NUMERIC_DATE_RE = re.compile(
    r"\b(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{4})\b"
)

# Matches relative temporal phrases to flag in tsvector
_RELATIVE_PHRASES = re.compile(
    r"\b(last\s+(?:week|month|year)|"
    r"(?:a\s+)?(?:few|couple\s+of)\s+(?:weeks|months|years)\s+ago|"
    r"(?:the\s+)?(?:previous|prior|following|next)\s+(?:week|month|year|day)|"
    r"(?:two|three|four|five|six|seven|eight|nine|ten)\s+(?:days?|weeks?|months?|years?)\s+(?:ago|later|before|after))\b",
    re.IGNORECASE,
)


def _emit_date_forms(year: int, month: int, day: int) -> list[str]:
    """Return all normalized forms of a date for tsvector injection."""
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100):
        return []
    mn = _MONTH_NAMES[month]
    ma = _MONTH_ABBR[month]
    forms = [
        f"{year}-{month:02d}-{day:02d}",           # ISO
        f"{mn} {day} {year}",                       # "january 1 2022"
        f"{ma} {day} {year}",                       # "jan 1 2022"
        f"{day} {mn} {year}",                       # "1 january 2022"
        f"{day} {ma} {year}",                       # "1 jan 2022"
        f"{month}/{day}/{year}",                    # US numeric
        f"{day}/{month}/{year}",                    # UK numeric
        f"{mn} {year}",                             # "january 2022" (month-level)
    ]
    return forms


def extract_date_terms(text: str) -> list[str]:
    """Extract and normalize all date expressions found in text.

    Returns a deduplicated list of normalized date strings suitable for
    injection into tsvector via boost_tsvector_terms().  The list is
    empty when no dates are found, so callers can skip the DB write.
    """
    terms: set[str] = set()

    # Named-month dates
    for m in _NAMED_DATE_RE.finditer(text):
        g = m.groupdict()
        if g.get("mn1"):
            mn_str = g["mn1"].lower()
            month = _MONTH_NAME_TO_NUM.get(mn_str[:3])
            day = int(g["d1"])
            year = int(g["y1"])
        else:
            mn_str = g["mn2"].lower()
            month = _MONTH_NAME_TO_NUM.get(mn_str[:3])
            day = int(g["d2"])
            year = int(g["y2"])
        if month:
            terms.update(_emit_date_forms(year, month, day))

    # ISO / numeric dates
    for m in _ISO_DATE_RE.finditer(text):
        g = m.groupdict()
        terms.update(_emit_date_forms(int(g["y"]), int(g["m"]), int(g["d"])))

    # Ambiguous numeric dates — emit both US (m/d/y) and UK (d/m/y) interpretations
    for m in _NUMERIC_DATE_RE.finditer(text):
        g = m.groupdict()
        a, b, y = int(g["a"]), int(g["b"]), int(g["y"])
        terms.update(_emit_date_forms(y, a, b))  # US: a=month, b=day
        terms.update(_emit_date_forms(y, b, a))  # UK: a=day, b=month

    # Relative phrases: emit as-is so FTS can match them
    for m in _RELATIVE_PHRASES.finditer(text):
        phrase = re.sub(r"\s+", " ", m.group(0).lower().strip())
        terms.add(phrase)

    # Remove single-character noise
    return [t for t in terms if len(t) > 1]
