"""Shared pytest fixtures for the pymacos test suite."""
import pytest


@pytest.fixture(scope="session")
def session_dir(tmp_path_factory):
    """Per-session scratch directory for tests that write a temp .in file.

    pytest cleans this up automatically when the session ends. Returns a
    pathlib.Path so callers can do `session_dir / "tempo.in"`.
    """
    return tmp_path_factory.mktemp("pymacos_session")
