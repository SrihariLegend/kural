"""Word features shared by training and the run-time: hashed character trigrams, a hashed whole word,
and a few shape flags. Pure Python and deterministic, so both sides see identical numbers.

Hashing means there is no vocabulary file and no unknown words: a file name the model has never
seen still gets a usable representation, and a typo shares most trigrams with the right spelling.
"""
N_TRI = 8192          # trigram buckets
N_WORD = 4096         # whole-word buckets
MAX_TRI = 20          # trigrams kept per word
N_SHAPE = 16
MAX_WORDS = 96         # request words plus the words of the session context


def _fnv(s):
    h = 2166136261
    for b in s.encode("utf-8"):
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return h


def word_features(w):
    """-> (trigram ids, word id, shape flags)"""
    low = w.lower()
    padded = f"<{low}>"
    tris = [_fnv(padded[i:i + 3]) % N_TRI for i in range(max(1, len(padded) - 2))][:MAX_TRI]
    digits = sum(c.isdigit() for c in w)
    shape = [
        digits > 0, digits == len(w), "/" in w, w.startswith("~"), "." in w, "*" in w or "?" in w,
        any(c.isupper() for c in w), w.isupper() and len(w) > 1, len(w) <= 1, 2 <= len(w) <= 3, 4 <= len(w) <= 7,
        len(w) >= 8, "-" in w or "_" in w, " " in w, w.startswith("."), bool(w) and w[0].isdigit() and not w.isdigit(),
    ]
    return tris, _fnv("w:" + low) % N_WORD, [1.0 if s else 0.0 for s in shape]
