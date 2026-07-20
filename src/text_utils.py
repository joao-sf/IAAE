from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any

SPACE_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]+")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = NON_ALNUM_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def parse_decimal(value: Any) -> float | None:
    """Converte números em formatos JSON, pt-BR ou en-US para float."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)

    text = str(value).strip().replace("R$", "").replace(" ", "")
    if not text:
        return None

    # Ex.: 1.234,56 -> 1234.56; 1,234.56 -> 1234.56; 1234,56 -> 1234.56
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        text = text.replace(".", "").replace(",", ".")

    try:
        return float(Decimal(text))
    except InvalidOperation, ValueError:
        return None
