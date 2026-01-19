import re


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_name(name: str) -> str:
    trimmed = name.strip()
    return _WHITESPACE_RE.sub(" ", trimmed)
