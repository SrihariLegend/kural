"""The Rust engine (engine/), used from Python through its C interface when it has been built.

run.parse uses it automatically when target/release/libkural_engine.so (or KURAL_ENGINE) and the model
file kural/model.kural exist; otherwise the NumPy code in run.py is used. KURAL_NATIVE=0 turns it off.
"""
import ctypes
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None
_ENGINE = None


def _paths():
    lib = os.environ.get("KURAL_ENGINE") or os.path.join(_HERE, "..", "target", "release", "libkural_engine.so")
    model = os.environ.get("KURAL_MODEL") or os.path.join(_HERE, "model.kural")
    return lib, model


def available():
    global _LIB, _ENGINE
    if _ENGINE is not None:
        return True
    if os.environ.get("KURAL_NATIVE") == "0":
        return False
    lib, model = _paths()
    if not (os.path.exists(lib) and os.path.exists(model)):
        return False
    try:
        L = ctypes.CDLL(lib)
    except OSError:
        return False
    L.kural_open.restype = ctypes.c_void_p
    L.kural_open.argtypes = [ctypes.c_char_p]
    L.kural_parse.restype = ctypes.c_int64
    L.kural_parse.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_char_p, ctypes.c_size_t]
    e = L.kural_open(model.encode())
    if not e:
        return False
    _LIB, _ENGINE = L, e
    return True


def parse(text, context=None, beam=4, keep=3):
    """Same result as run.parse: list of (program, probability, request words, copied positions)."""
    ctx = "\x1f".join(context or []).encode()
    cap = 1 << 16
    while True:
        buf = ctypes.create_string_buffer(cap)
        n = _LIB.kural_parse(_ENGINE, text.encode(), ctx, beam, keep, buf, cap)
        if n < 0:
            return []
        if n < cap:
            break
        cap = n + 1
    out = []
    for r in json.loads(buf.value.decode()):
        program = [[(name, args) for name, args in pipe] for pipe in r["program"]]
        out.append((program, r["prob"], r["words"], set(r["used"])))
    return out
