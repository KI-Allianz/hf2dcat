import os
import re
import pytest
from typer.testing import CliRunner
from hf2dcat.cli import app

@pytest.fixture(scope="session")
def runner():
    return CliRunner()

@pytest.fixture(scope="session")
def prog_name():
    return "hf2dcat"

@pytest.fixture(scope="session")
def app_obj():
    return app

@pytest.fixture
def vcr_config():
    return {
        "cassette_library_dir": "tests/cassettes",
        "record_mode": "once", 
    }