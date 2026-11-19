"""8.3: подпись Ed25519 (RFC 8032) без внешних зависимостей — для манифеста обновлений.

Код повторяет эталонную реализацию из RFC 8032, раздел 6, и сверяется с тестовыми
векторами RFC (`tests/test_updates.py`). Проверка одной подписи занимает доли секунды —
этого достаточно для манифеста. Подписывает выпуск (`sign`) только машина сборки с
закрытым ключом; приложение лишь проверяет (`verify`) открытым ключом.
"""

import hashlib

_P = 2**255 - 19
_D = -121665 * pow(121666, _P - 2, _P) % _P
_Q = 2**252 + 27742317777372353535851937790883648493
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)

Point = tuple[int, int, int, int]


def _sha512_modq(data: bytes) -> int:
    return int.from_bytes(hashlib.sha512(data).digest(), "little") % _Q


def _add(a: Point, b: Point) -> Point:
    x1 = (a[1] - a[0]) * (b[1] - b[0]) % _P
    x2 = (a[1] + a[0]) * (b[1] + b[0]) % _P
    c = 2 * a[3] * b[3] * _D % _P
    d = 2 * a[2] * b[2] % _P
    e, f, g, h = x2 - x1, d - c, d + c, x2 + x1
    return (e * f, g * h, f * g, e * h)


def _mul(scalar: int, point: Point) -> Point:
    result: Point = (0, 1, 1, 0)
    while scalar > 0:
        if scalar & 1:
            result = _add(result, point)
        point = _add(point, point)
        scalar >>= 1
    return result


def _equal(a: Point, b: Point) -> bool:
    return (a[0] * b[2] - b[0] * a[2]) % _P == 0 and (a[1] * b[2] - b[1] * a[2]) % _P == 0


def _recover_x(y: int, sign: int) -> int | None:
    if y >= _P:
        return None
    x2 = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P)
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (_P + 3) // 8, _P)
    if (x * x - x2) % _P != 0:
        x = x * _SQRT_M1 % _P
    if (x * x - x2) % _P != 0:
        return None
    if (x & 1) != sign:
        x = _P - x
    return x


_GY = 4 * pow(5, _P - 2, _P) % _P
_GX = _recover_x(_GY, 0)
assert _GX is not None
_G: Point = (_GX, _GY, 1, _GX * _GY % _P)


def _compress(point: Point) -> bytes:
    zinv = pow(point[2], _P - 2, _P)
    x, y = point[0] * zinv % _P, point[1] * zinv % _P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _decompress(data: bytes) -> Point | None:
    if len(data) != 32:
        return None
    y = int.from_bytes(data, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _recover_x(y, sign)
    return None if x is None else (x, y, 1, x * y % _P)


def _expand(secret: bytes) -> tuple[int, bytes]:
    if len(secret) != 32:
        raise ValueError("Ed25519 secret key must be 32 bytes")
    digest = hashlib.sha512(secret).digest()
    scalar = int.from_bytes(digest[:32], "little")
    scalar &= (1 << 254) - 8
    scalar |= 1 << 254
    return scalar, digest[32:]


def public_key(secret: bytes) -> bytes:
    """Открытый ключ (32 байта) из закрытого (32 байта)."""
    scalar, _ = _expand(secret)
    return _compress(_mul(scalar, _G))


def sign(secret: bytes, message: bytes) -> bytes:
    """Подпись (64 байта) — только для машины сборки выпуска."""
    scalar, prefix = _expand(secret)
    public = _compress(_mul(scalar, _G))
    r = _sha512_modq(prefix + message)
    encoded_r = _compress(_mul(r, _G))
    h = _sha512_modq(encoded_r + public + message)
    s = (r + h * scalar) % _Q
    return encoded_r + int.to_bytes(s, 32, "little")


def verify(public: bytes, message: bytes, signature: bytes) -> bool:
    """Подпись `signature` сообщения `message` сделана закрытым ключом к `public`."""
    if len(public) != 32 or len(signature) != 64:
        return False
    a = _decompress(public)
    r = _decompress(signature[:32])
    if a is None or r is None:
        return False
    s = int.from_bytes(signature[32:], "little")
    if s >= _Q:
        return False
    h = _sha512_modq(signature[:32] + public + message)
    return _equal(_mul(s, _G), _add(r, _mul(h, a)))
