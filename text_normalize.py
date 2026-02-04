import os
import re
from typing import Callable


_RE_NUM = re.compile(r"\d+(?:\.\d+)?%?")


_DIGIT_MAP = {
    "0": "零",
    "1": "一",
    "2": "二",
    "3": "三",
    "4": "四",
    "5": "五",
    "6": "六",
    "7": "七",
    "8": "八",
    "9": "九",
}


def _digits_to_zh(s: str) -> str:
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in s)


def normalize_numbers_zh(text: str) -> str:
    """Convert Arabic numerals to Chinese-friendly spoken text.

    Heuristics (good enough for TTS):
    - Long sequences (>=7) or leading zeros: speak digit-by-digit (e.g. phone numbers).
    - 4-digit years like 2026: speak digit-by-digit (二零二六).
    - Decimals: '3.14' -> '三点一四'
    - Percent: '50%' -> '百分之五十'
    - Others: use cn2an.an2cn (e.g. 123 -> 一百二十三)
    """
    s = (text or "")
    if not s:
        return s

    try:
        import cn2an  # type: ignore
    except Exception:
        # Fallback: digit-by-digit only
        def _rep(m):
            raw = m.group(0)
            if raw.endswith("%"):
                return "百分之" + _digits_to_zh(raw[:-1])
            if "." in raw:
                a, b = raw.split(".", 1)
                return _digits_to_zh(a) + "点" + _digits_to_zh(b)
            return _digits_to_zh(raw)

        return _RE_NUM.sub(_rep, s)

    def _rep(m):
        raw = m.group(0)
        is_percent = raw.endswith("%")
        num = raw[:-1] if is_percent else raw

        # Decimal
        if "." in num:
            a, b = num.split(".", 1)
            left = cn2an.an2cn(a, "low") if a and not (len(a) > 1 and a.startswith("0")) else _digits_to_zh(a)
            right = _digits_to_zh(b)
            out = f"{left}点{right}"
            return ("百分之" + out) if is_percent else out

        # Integers
        if len(num) > 1 and num.startswith("0"):
            out = _digits_to_zh(num)
            return ("百分之" + out) if is_percent else out

        if len(num) >= 7:
            out = _digits_to_zh(num)
            return ("百分之" + out) if is_percent else out

        # Year-like 4 digits
        if len(num) == 4:
            try:
                val = int(num)
                if 1000 <= val <= 2099:
                    out = _digits_to_zh(num)
                    return ("百分之" + out) if is_percent else out
            except Exception:
                pass

        try:
            out = cn2an.an2cn(num, "low")
        except Exception:
            out = _digits_to_zh(num)

        if is_percent:
            return "百分之" + out
        return out

    return _RE_NUM.sub(_rep, s)


def strip_markdown_for_tts(text: str) -> str:
    """Make LLM text more "speakable" for TTS engines.

    - Remove code fences/backticks
    - Remove markdown emphasis markers (*, **, __)
    - Replace some bullets with Chinese pauses
    """
    s = (text or "")
    if not s:
        return s

    # Code fences and inline code markers
    s = s.replace("```", "")
    s = s.replace("`", "")

    # Emphasis markers
    for token in ("**", "__", "*"):
        s = s.replace(token, "")

    # Common bullet styles
    s = s.replace("- ", "，")
    s = s.replace("•", "，")

    # Remove extra brackets that often appear in markdown
    s = s.replace("[", "").replace("]", "")

    # Collapse excessive whitespace
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_for_zh_tts(text: str) -> str:
    s = strip_markdown_for_tts(text)
    s = normalize_numbers_zh(s)

    # Optional: remove emojis / rare symbols (keep common Chinese punctuation)
    if os.environ.get("TTS_STRIP_NON_TEXT", "1").strip().lower() in ("1", "true", "yes"):
        # Keep: CJK, ASCII letters/digits (digits already normalized), spaces, and common punctuation
        s = re.sub(r"[^\u4e00-\u9fffA-Za-z零一二三四五六七八九十百千万亿点负%。，！？；：、“”‘’（）()《》<>【】,.;:!? \n\r\t-]", "", s)

    return s.strip()

