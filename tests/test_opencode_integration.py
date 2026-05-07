"""Regression tests for the OpenCode integration source."""

from __future__ import annotations

from pathlib import Path


def test_opencode_policy_includes_temp_allow_paths() -> None:
    """OpenCode should match other integrations by allowing OS temp dirs."""
    source = Path("integrations/opencode/intaris.ts").read_text()

    assert '"/tmp/*"' in source
    assert '"/var/tmp/*"' in source
    assert "process.env.TMPDIR" in source
    assert "transient scratch" in source
