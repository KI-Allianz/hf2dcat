import pytest
from pathlib import Path

@pytest.mark.vcr
def test_fetch_include_restricted_vcr(runner, app_obj, prog_name, tmp_path: Path):
    res = runner.invoke(
        app_obj, ["fetch", "-t", "dataset", "-l", "2", "-R", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.json"))
