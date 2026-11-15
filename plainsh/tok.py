"""Splitting a request into words. The same function is used to make training data and at run time,
so word positions always mean the same thing."""
import re

# a quote only opens at the start of a word and closes at the end of one, so "what's" and "don't" are left alone
_QUOTED = re.compile(r"""(?<![\w'])(["'])(.+?)\1(?![\w'])""")
_TRAIL = ",;!?"


def words(text):
    """'delete "old stuff" in ~/tmp, please.' -> ['delete', 'old stuff', 'in', '~/tmp', 'please']

    Quoted text becomes one word (quotes removed). Trailing sentence punctuation is dropped,
    but dots inside names (file.txt, *.log, v1.2) are kept.
    """
    out, pos = [], 0
    for m in _QUOTED.finditer(text):
        out += _plain(text[pos:m.start()])
        out.append(m.group(2))
        pos = m.end()
    out += _plain(text[pos:])
    return out


def _plain(chunk):
    res = []
    for w in chunk.split():
        w = w.rstrip(_TRAIL)
        if len(w) > 1 and w.endswith(".") and not w.endswith(".."):
            w = w[:-1]
        w = w.lstrip("(").rstrip(")")
        if w:
            res.append(w)
    return res
