"""
nlp_layer.py — Sentence construction from a stream of gesture tokens.

Responsibilities
────────────────
- De-duplicate: collapse consecutive identical gestures into one token.
- Map raw labels to natural words/phrases  (e.g. "thank_you" → "thank you").
- Assemble tokens into a readable sentence string.
- Expose a `SentenceBuffer` class that  inference.py  drives.

This is intentionally kept simple for Phase 4.
The "Next steps" section of README describes adding a small transformer
polishing pass over the output — that would plug in here.
"""

from collections import deque
from typing import Optional


# ── Token → surface-form mapping ──────────────────────────────────────────────
# Add / edit entries freely.  The key is the raw label from GESTURE_LABELS,
# the value is what gets spoken aloud / displayed.
_TOKEN_MAP: dict[str, str] = {
    "hello":       "hello",
    "yes":         "yes",
    "no":          "no",
    "please":      "please",
    "thank_you":   "thank you",
    "i_love_you":  "I love you",
    "peace":       "peace",
    "thumbs_up":   "good",
    "thumbs_down": "not good",
    "stop":        "stop",
}


def token_to_word(token: str) -> str:
    """Map a raw gesture label to its display/speech surface form."""
    return _TOKEN_MAP.get(token, token.replace("_", " "))


class SentenceBuffer:
    """
    Collects gesture tokens and builds a sentence string.

    Typical usage in inference.py
    ──────────────────────────────
        buf = SentenceBuffer(max_tokens=20)
        buf.push("hello")           # → internal buffer updated
        buf.push("hello")           # → ignored (de-dup)
        buf.push("please")          # → new token added
        sentence = buf.flush()      # → "hello please"  and buffer cleared
    """

    def __init__(self, max_tokens: int = 20) -> None:
        self._tokens: deque[str] = deque(maxlen=max_tokens)
        self._last_token: Optional[str] = None

    def push(self, token: str) -> bool:
        """
        Add a token if it differs from the previous one.

        Returns True if the token was actually added (i.e. it was new).
        """
        if token == self._last_token:
            return False
        self._tokens.append(token_to_word(token))
        self._last_token = token
        return True

    def flush(self) -> str:
        """
        Return the assembled sentence and clear the buffer.

        After flushing, `last_token` is also reset so the very next
        push is always accepted.
        """
        sentence = " ".join(self._tokens).strip()
        self._tokens.clear()
        self._last_token = None
        return sentence

    def peek(self) -> str:
        """Return the current (unflushed) sentence without clearing it."""
        return " ".join(self._tokens).strip()

    def __len__(self) -> int:
        return len(self._tokens)

    def is_empty(self) -> bool:
        return len(self._tokens) == 0
