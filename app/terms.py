import re
import unicodedata


def normalize_term(term: str) -> str:
    return unicodedata.normalize("NFC", term).casefold().strip()


def _needs_plain_matching(term: str) -> bool:
    for char in term:
        if char.isalpha() and "LATIN" not in unicodedata.name(char, ""):
            return True
    return False


def contains_term(text: str, term: str) -> bool:
    """Match CJK/Korean/etc. as text, and Latin terms as whole tokens."""
    if not term: return False
    text = unicodedata.normalize("NFC", text)
    term = unicodedata.normalize("NFC", term)
    if _needs_plain_matching(term): return normalize_term(term) in normalize_term(text)
    return re.search(r"(?<![A-Za-z0-9_])" + re.escape(term) + r"(?![A-Za-z0-9_])", text, re.IGNORECASE) is not None


def contains_any(text: str, terms: list[str]) -> bool:
    return any(contains_term(text, term) for term in terms)


def replace_terms(text: str, replacements: dict[str, str]) -> str:
    for source in sorted(replacements, key=len, reverse=True):
        target = replacements[source]
        if not source: continue
        if _needs_plain_matching(source):
            text = text.replace(source, target)
        else:
            text = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(source) + r"(?![A-Za-z0-9_])", target, text, flags=re.IGNORECASE)
    return text
