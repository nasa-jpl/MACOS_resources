# conftest.py
import pytest
from pathlib import Path
from context import PRJ_PATH


@pytest.fixture(scope="session")
def session_dir(tmp_path_factory):
    # Use tmp_path_factory to create a temporary directory inside the custom parent
    # The factory will handle the teardown and cleanup at the end of the session.
    temp_dir: Path = tmp_path_factory.mktemp("data")

    return temp_dir

