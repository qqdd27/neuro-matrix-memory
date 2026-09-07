"""Contract tests: provider lifecycle the way MemoryManager drives it.

Run:  python tests/test_contract.py
Covers: initialize -> sync_turn xN -> prefetch -> tools -> on_session_end
(sleep + rollup) -> on_pre_compress -> on_memory_write -> shutdown, plus a
concurrent-write smoke (MemoryManager background threads).
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuro_matrix.provider import NeuromatrixMemoryProvider, NM_TOOL_SCHEMA  # noqa: E402


def test_full_lifecycle():
    tmp = os.path.join(tempfile.mkdtemp(), "contract")
    os.makedirs(tmp, exist_ok=True)
    db = os.path.join(tmp, "nm.db")
    p = NeuromatrixMemoryProvider(config={
        "db_path": db, "auto_consolidate": "true", "llm_enabled": "false",
        "retention_days": "30", "max_recall_chars": "600",
    })
    # MemoryManager contract order.
    assert p.is_available() is True
    assert p.name == "neuromatrix"
    assert p.get_config_schema() and p.get_config_schema()[0]["key"] == "db_path"
    p.initialize("session-1", hermes_home=tmp, agent_context="primary",
                 platform="test", user_id="u1")
    assert p.system_prompt_block().startswith("# NeuroMatrix Memory")
    assert p.get_tool_schemas() and p.get_tool_schemas()[0]["name"] == "neuromatrix"
    # Several turns across two sessions.
    p.sync_turn("Какой статус у id_101?", "id_101: активен, выплаты идут.", session_id="sessA")
    p.sync_turn("Обнови id_101.", "id_101: баланс вырос.", session_id="sessA")
    p.sync_turn("Проверь id_777.", "id_777 — это TON кошелёк.", session_id="sessB")
    # Prefetch injects bounded recall.
    recall = p.prefetch("id_777", session_id="sessB")
    assert isinstance(recall, str) and "TON" in recall
    status = p.recall_status()
    assert status is not None and status.provider_label == "NeuroMatrix"
    # Tool round-trips.
    import json
    r = json.loads(p.handle_tool_call("neuromatrix", {"action": "probe", "entity": "id_777"}))
    assert r["ok"] and r["found"] and ("ton" in (r.get("key") or "").lower()
                                       or "ton" in [a.lower() for a in r.get("aliases", [])]), r
    # Session end: rollup + sleep + prune all run without error.
    p.on_session_end([{"role": "user", "content": "какой статус id_101?"},
                      {"role": "assistant", "content": "id_101 активен."}])
    # MemoryManager switches session: hook must not raise.
    p.on_session_switch("session-2", reset=True)
    p.sync_turn("Новая сессия про id_777.", "id_777 снова на связи.", session_id="session-2")
    # Pre-compress evidence archive.
    out = p.on_pre_compress([{"role": "user", "content": "проверь id_777 ещё раз"}])
    assert isinstance(out, str)
    # Built-in memory mirror.
    p.on_memory_write("add", "user", "Owner prefers fact-based answers.", None)
    # Shutdown closes cleanly.
    p.shutdown()
    print("lifecycle OK")


def test_concurrent_writes():
    """sync_turn may run on MemoryManager background threads while prefetch
    runs on the agent thread — the store lock must serialize them."""
    tmp = os.path.join(tempfile.mkdtemp(), "contract2")
    db = os.path.join(tmp, "nm.db")
    p = NeuromatrixMemoryProvider(config={"db_path": db, "llm_enabled": "false"})
    p.initialize("s", hermes_home=tmp, agent_context="primary")
    errors: list[Exception] = []

    def writer(i: int):
        try:
            for j in range(10):
                p.sync_turn(f"Как id_{i}?",
                            f"id_{i}: статус {j}, всё ок.", session_id=f"w{i}")
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for _ in range(20):
        p.prefetch("id_777")  # reads during writes
    for t in threads:
        t.join()
    assert not errors, errors
    p.shutdown()
    print("concurrency OK")


def _run_all() -> None:
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _run_all()
