from __future__ import annotations

import pytest

from a100_agent_lab.sessions import Session, SessionStore


def test_session_store_create_get_list_reset_delete() -> None:
    store = SessionStore()

    first = store.create(system_prompt="First system prompt.")
    second = store.create(system_prompt="Second system prompt.")

    assert first.id != second.id
    assert store.get(first.id) is first
    assert store.get(second.id) is second
    assert [session.id for session in store.list()] == [first.id, second.id]

    first.add_user_message("Question")
    first.add_assistant_message("Answer")

    assert first.turn_count == 1
    assert first.updated_at >= first.created_at

    reset = store.reset(first.id)

    assert reset is first
    assert first.turn_count == 0
    assert first.transcript() == [{"role": "system", "content": "First system prompt."}]
    assert store.delete(first.id) is True
    assert store.delete(first.id) is False
    assert [session.id for session in store.list()] == [second.id]

    with pytest.raises(KeyError):
        store.get(first.id)


def test_session_store_add_existing_session_and_clear() -> None:
    store = SessionStore()
    session = Session(system_prompt="External session.")

    assert store.add(session) is session
    assert store.get(session.id) is session

    store.clear()

    assert store.list() == []
