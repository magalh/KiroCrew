"""Turn off git's config-defined hooks for one host-side git call.

Git 2.54 runs hooks named in config, not just hook files::

    [hook "x"]
        command = <any shell command>
        event = pre-commit

``-c core.hooksPath=<devnull>`` only moves the hook DIRECTORY, so these still
run. A repository's own ``.git/config`` is writable by whoever edits the tree,
so a host-side git call over an agent-writable tree would run that command
outside the sandbox. They fire on ordinary porcelain: ``add``, ``commit``,
``checkout``, ``worktree add``, ``push``, ``fetch``, ``update-ref``, and even
``status`` when it refreshes the index (``post-index-change``).

There is no switch that turns them all off. ``hook.<event>.enabled=false``
(git 2.55+) is ignored when the same ``<event>`` is also used as a hook name,
which the repository can arrange. ``hook.<name>.enabled=false`` works on every
version that has config hooks, but only for a name we know. So each call first
lists the names git will see and disables each one by name. Older git ignores
the extra keys.

Every scope is disabled, not just the repository's: the callers already point
``core.hooksPath`` at the null device, so the user's own hooks never ran on
these calls either.

The list is read just before the call, by a separate process, so a writer that
is running at the same moment can add a new name in between. That is the same
limit the ``.git/info/attributes`` pin in
:mod:`kiro_crew.apps.builtins.auto_improvement.spine.git_safety` has.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping

from kiro_crew import platform_compat

__all__ = [
    "MAX_HOOK_NAMES",
    "ConfigHookScanError",
    "config_hook_disable_args",
    "config_hook_names",
]

#: More names than this is refused rather than passed on: no real setup has this
#: many, and each one costs two argv entries.
MAX_HOOK_NAMES = 256

#: Individual names longer than this (in UTF-8 bytes) cannot be passed as ``-c``
#: arguments on any realistic platform without risking E2BIG. Reject them early
#: so a single enormous name in the repository config cannot crash a call site.
_MAX_HOOK_NAME_BYTES = 512

_SCAN_TIMEOUT_SECS = 10

#: Bound once, so a stand-in a caller's tests install for ``subprocess.run`` (to fake
#: that caller's own git) does not also answer this scan.
_run = subprocess.run


class ConfigHookScanError(RuntimeError):
    """The hook names could not be listed safely, so the git call must not run."""


def _resolve_git(git: str) -> str | None:
    """Resolve a git executable name to a trusted path.

    When *git* is the literal ``"git"`` default, resolve through the trusted
    system directories (``platform_compat.trusted_git_bin``). A planted shim on
    an agent-writable PATH entry would otherwise run here with the gateway's full
    credentials, because the scan subprocess inherits the unfiltered environment
    before ``sandboxed_spawn_argv`` can strip it.

    A caller that already holds a trusted absolute path passes it directly, and
    it is returned unchanged. ``None`` means the binary cannot be resolved: the
    caller should treat it as an empty scan rather than fall back to a bare name.
    """
    if git == "git":
        return platform_compat.trusted_git_bin()
    return git


def config_hook_names(
    cwd: str | os.PathLike[str],
    *,
    git: str = "git",
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Every ``hook.<name>.*`` name git would read in ``cwd``, from all scopes.

    ``git`` and ``env`` should be what the real call uses, so both find the same
    binary and see the same config files. A ``cwd`` that is not a directory, or a
    ``git`` that cannot be found in a trusted location, returns ``[]``: the real
    call fails on its own there.

    Raises :class:`ConfigHookScanError` when the listing fails, when a name
    contains ``=`` (``-c`` splits on the first ``=``, so it could not be
    disabled), when any name exceeds :data:`_MAX_HOOK_NAME_BYTES`, or when there
    are more than :data:`MAX_HOOK_NAMES` names.
    """
    if not os.path.isdir(cwd):
        return []
    resolved = _resolve_git(git)
    if resolved is None:
        # git is not in a trusted location; the real call would also fail.
        return []
    argv = [
        resolved,
        "-C",
        os.fspath(cwd),
        "-c",
        "core.fsmonitor=false",
        "config",
        "-z",
        "--name-only",
        "--get-regexp",
        r"^hook\.",
    ]
    try:
        proc = _run(
            argv,
            env=dict(env) if env is not None else None,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=_SCAN_TIMEOUT_SECS,
            check=False,
        )
    except FileNotFoundError:
        # No such git under this name and PATH: the real call, which spawns the same
        # name with the same environment, cannot run either, so there is nothing to
        # disable.
        return []
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigHookScanError(f"could not list git hook config: {exc}") from exc
    # 0 = keys found, 1 = no matching key. Anything else is an unreadable config,
    # which the real call would also choke on; refuse rather than guess.
    if proc.returncode not in (0, 1):
        tail = proc.stderr.decode("utf-8", "replace").strip()[-200:]
        raise ConfigHookScanError(f"could not list git hook config ({proc.returncode}): {tail}")
    names: list[str] = []
    seen: set[str] = set()
    for raw in proc.stdout.split(b"\0"):
        # surrogateescape so a non-UTF-8 name goes back to git byte-for-byte.
        key = raw.decode("utf-8", "surrogateescape")
        if not key.startswith("hook."):
            continue
        rest = key[len("hook.") :]
        if "." not in rest:
            continue  # `hook.jobs`: a setting, not a named hook
        name = rest[: rest.rindex(".")]
        if name in seen:
            continue
        if "=" in name:
            raise ConfigHookScanError(
                f"git hook name {name[:80]!r} contains '=' and cannot be disabled"
            )
        name_bytes = name.encode("utf-8", "surrogateescape")
        if len(name_bytes) > _MAX_HOOK_NAME_BYTES:
            raise ConfigHookScanError(
                f"git hook name is {len(name_bytes)} bytes (limit {_MAX_HOOK_NAME_BYTES})"
            )
        seen.add(name)
        names.append(name)
    if len(names) > MAX_HOOK_NAMES:
        raise ConfigHookScanError(
            f"git config defines {len(names)} hook names (limit {MAX_HOOK_NAMES})"
        )
    return names


def config_hook_disable_args(
    cwd: str | os.PathLike[str],
    *,
    git: str = "git",
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """``-c hook.<name>.enabled=false`` argv entries; place them before the subcommand.

    Empty when no hook is configured. Raises like :func:`config_hook_names`.
    """
    args: list[str] = []
    for name in config_hook_names(cwd, git=git, env=env):
        args += ["-c", f"hook.{name}.enabled=false"]
    return args
