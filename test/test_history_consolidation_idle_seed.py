"""The idle sweep must find sessions left unconsolidated before a restart.

``HistoryConsolidator._last_activity`` is in memory only and was written only by
``maybe_consolidate``. After a gateway restart, a session with an unconsolidated
tail that nobody touched again was never in that dict, so ``check_idle_sessions``
never looked at it and its tail was never consolidated. The first sweep now seeds
the dict once, off the event loop, from the transcripts on disk.
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kiro_crew import history as history_mod
from kiro_crew.history import ConversationLog, HistoryConsolidator

STEM = "dashboard_chat-left-behind"


def _restarted(
    tmp_path, *, unconsolidated: bool, memory_mode: str = ""
) -> tuple[ConversationLog, HistoryConsolidator]:
    """A transcript written by an earlier process, read by a fresh consolidator."""
    writer = ConversationLog(base_dir=tmp_path / "sessions")
    writer.init()
    with history_mod.allow_on_loop_persist():
        for i in range(3):
            writer.append("dashboard:chat-left-behind", "user", f"m{i}")
        if not unconsolidated:
            writer.mark_consolidated("dashboard:chat-left-behind", 3)
        if memory_mode:
            writer.update_metadata("dashboard:chat-left-behind", {"memory_mode": memory_mode})
    old = time.time() - 7 * 86400
    os.utime(tmp_path / "sessions" / f"{STEM}.jsonl", (old, old))
    log = ConversationLog(base_dir=tmp_path / "sessions")
    memory = MagicMock()
    memory.read_preferences.return_value = ""
    memory.read_projects.return_value = ""
    return log, HistoryConsolidator(log=log, memory=memory, migrated=True, history_idle_secs=3600)


async def _sweep_after_seed(consolidator: HistoryConsolidator) -> list[str]:
    started: list[str] = []

    async def fake_consolidate(key: str, include_history: bool = True) -> None:
        started.append(key)
        consolidator._running.discard(key)

    with patch.object(consolidator, "_consolidate", AsyncMock(side_effect=fake_consolidate)):
        consolidator.check_idle_sessions()  # starts the one-time seed
        seed = getattr(consolidator, "_seed_task", None)
        if seed is not None:
            await seed
        consolidator.check_idle_sessions()
        await asyncio.gather(*list(consolidator._tasks), return_exceptions=True)
    return started


@pytest.mark.asyncio
async def test_untouched_unconsolidated_session_is_picked_up_after_restart(tmp_path):
    _log, consolidator = _restarted(tmp_path, unconsolidated=True)

    started = await _sweep_after_seed(consolidator)

    assert started == [STEM]
    # Seeded from the file's mtime, not "now": the session is already idle.
    assert consolidator._last_activity[STEM] < time.time() - 86400


@pytest.mark.asyncio
async def test_fully_consolidated_session_is_not_seeded(tmp_path):
    _log, consolidator = _restarted(tmp_path, unconsolidated=False)

    consolidator.check_idle_sessions()
    assert consolidator._seed_task is not None
    await consolidator._seed_task

    assert consolidator._last_activity == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["incognito", "temporary"])
async def test_private_transcript_is_not_seeded(tmp_path, mode):
    """``_consolidate`` refuses it, and a refusal sets no throttle: seeding it
    would re-dispatch a refused task on every sweep."""
    _log, consolidator = _restarted(tmp_path, unconsolidated=True, memory_mode=mode)

    assert await _sweep_after_seed(consolidator) == []
    assert consolidator._last_activity == {}


@pytest.mark.asyncio
async def test_session_end_does_not_start_a_second_run_under_the_live_key(tmp_path):
    """A seeded stem already running blocks the same transcript's live key."""
    _log, consolidator = _restarted(tmp_path, unconsolidated=True)
    consolidator._running.add(STEM)

    with patch.object(consolidator, "_consolidate", AsyncMock()) as run:
        consolidator.consolidate_session("dashboard:chat-left-behind")
        consolidator.maybe_consolidate("dashboard:chat-left-behind")

    run.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_slack_file_and_canonical_key_share_one_run(tmp_path):
    """``slack:<ts>`` still reads a pre-migration ``<ts>.jsonl``: same transcript."""
    log, consolidator = _restarted(tmp_path, unconsolidated=True)
    (tmp_path / "sessions" / "1720000000.000001.jsonl").write_text("{}\n", encoding="utf-8")
    consolidator._running.add("1720000000.000001")

    assert consolidator._busy("slack:1720000000.000001")
    assert not consolidator._busy("slack:1720000000.000002")


@pytest.mark.asyncio
async def test_a_refused_seed_is_dropped_not_retried_every_sweep(tmp_path):
    """A seed made private after the scan is refused once, then forgotten."""
    from kiro_crew.history_consolidation import _CONSOLIDATION_REFUSED

    _log, consolidator = _restarted(tmp_path, unconsolidated=True)
    consolidator._activity_seeded = True
    consolidator._last_activity[STEM] = 0.0
    consolidator._seeded_keys.add(STEM)

    with patch.object(consolidator, "_consolidate", AsyncMock(return_value=_CONSOLIDATION_REFUSED)):
        consolidator.check_idle_sessions()
        await asyncio.gather(*list(consolidator._tasks), return_exceptions=True)

    assert STEM not in consolidator._last_activity


@pytest.mark.asyncio
async def test_live_key_replaces_the_seeded_alias(tmp_path):
    """Once the session speaks again it is tracked once, under its live key."""
    _log, consolidator = _restarted(tmp_path, unconsolidated=True)
    consolidator._activity_seeded = True
    consolidator._last_activity[STEM] = 0.0
    consolidator._seeded_keys.add(STEM)

    consolidator.maybe_consolidate("dashboard:chat-left-behind")

    assert list(consolidator._last_activity) == ["dashboard:chat-left-behind"]
