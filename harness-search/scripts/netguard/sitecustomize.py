"""Outbound-network guard for the offline re-synthesis fidelity probe.

Auto-imported by CPython when this directory is on PYTHONPATH. Every outbound
socket.connect is refused and appended to $SQ_NETBLOCK_LOG, so the d0 gate can count
attempts from a file the probed process itself wrote — not from a claim.

This is the deterministic form of "the roll-up performs no retrieval". The claim is
checkable because create_chat_completion appears exactly twice in tree_research.py, both
inside research_node / generate_child_questions — the assembly path below them touches no
network at all. If that stops being true the probe crashes loudly instead of quietly
buying search results.

Loopback is left open: the harness itself does not need it, but blocking it turns an
unrelated local service into a confusing crash rather than a clear verdict.
"""
from __future__ import annotations

import os
import socket

_LOG = os.environ.get("SQ_NETBLOCK_LOG", "")
_LOCAL = {"127.0.0.1", "::1", "localhost"}
_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _note(addr: object) -> None:
    if not _LOG:
        return
    try:
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(f"BLOCKED {addr!r}\n")
    except OSError:
        pass


def _host(address: object) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return ""


def _connect(self, address):  # type: ignore[no-untyped-def]
    if _host(address) in _LOCAL:
        return _real_connect(self, address)
    _note(address)
    raise OSError("SQ_NETGUARD: outbound network blocked during offline re-synthesis "
                  f"(attempted {address!r}). The roll-up must not retrieve.")


def _connect_ex(self, address):  # type: ignore[no-untyped-def]
    if _host(address) in _LOCAL:
        return _real_connect_ex(self, address)
    _note(address)
    return 111  # ECONNREFUSED


socket.socket.connect = _connect          # type: ignore[method-assign]
socket.socket.connect_ex = _connect_ex    # type: ignore[method-assign]
