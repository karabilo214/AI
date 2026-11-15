import sqlite3

import pytest

from ramvi_service.client_keys import KeyStore


@pytest.fixture
def store(tmp_path):
    s = KeyStore(tmp_path / "keys.db")
    yield s
    s.close()


def test_create_verify_counts_and_last_seen(store):
    rec, key = store.create("Claude Code", {"chat", "models"})
    assert rec.requests == 0 and rec.last_seen is None
    for _ in range(3):
        assert store.verify(key, "chat").id == rec.id
    after = store.get(rec.id)
    assert after.requests == 3 and after.last_seen is not None


def test_only_hash_stored_and_key_not_shown_again(store, tmp_path):
    rec, key = store.create("Codex", {"chat"})
    raw = (tmp_path / "keys.db").read_bytes() + (tmp_path / "keys.db-wal").read_bytes()
    assert key.encode() not in raw
    assert key not in repr(store.list()) and key not in repr(store.get(rec.id))


def test_separate_keys_permissions_and_revoke(store):
    a, key_a = store.create("A", {"chat"})
    b, key_b = store.create("B", {"models"})
    assert key_a != key_b
    assert store.verify(key_a, "models") is None  # нет права
    assert store.verify(key_b, "models").name == "B"
    store.revoke(a.id)
    store.revoke(a.id)  # повторно — без ошибки
    assert store.verify(key_a) is None and store.verify(key_b) is not None
    assert store.get(a.id).revoked is not None
    with pytest.raises(KeyError):
        store.revoke("missing")


@pytest.mark.parametrize("bad", [None, "", "ramvi-wrong", "sk-something"])
def test_bad_keys_rejected(store, bad):
    store.create("A", {"chat"})
    assert store.verify(bad) is None


def test_validation(store):
    with pytest.raises(ValueError):
        store.create(" ", {"chat"})
    with pytest.raises(ValueError):
        store.create("A", {"admin"})
    with pytest.raises(ValueError):
        store.create("A", set())


def test_survives_restart(tmp_path):
    s = KeyStore(tmp_path / "k.db")
    rec, key = s.create("A", {"chat"})
    s.close()
    s2 = KeyStore(tmp_path / "k.db")
    assert s2.verify(key).id == rec.id
    s2.close()
    assert sqlite3.connect(tmp_path / "k.db").execute("PRAGMA journal_mode").fetchone()[0] == "wal"


def test_route_permissions(store):
    from ramvi_service.client_keys import checker, permission_for

    assert permission_for("/v1/chat/completions") == "chat"
    assert permission_for("/api/generate?x=1") == "chat"
    assert permission_for("/v1/models") == "models" and permission_for("/engine/status") == "models"
    _, models_only = store.create("Монитор", {"models"})
    _, chat = store.create("Чат", {"chat", "models"})
    check = checker(store)
    assert check(models_only, "/v1/models") and not check(models_only, "/v1/messages")
    assert check(chat, "/v1/messages") and not check("ramvi-bad", "/v1/models")
