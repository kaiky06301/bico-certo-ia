"""Pré-processamento do texto digitado pelo morador."""
import re
import unicodedata


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos e sem pontuação: "Tá VAZANDO!" -> "ta vazando"."""
    texto = unicodedata.normalize("NFKD", (texto or "").lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-z0-9 ]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()
