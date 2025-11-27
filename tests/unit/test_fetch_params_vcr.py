import pytest
from pathlib import Path

@pytest.mark.vcr
@pytest.mark.parametrize("sort_key", ["likes", "likes7d"])
def test_fetch_sorted_by_params_vcr(runner, app_obj, prog_name, tmp_path: Path, sort_key):
    res = runner.invoke(
        app_obj,
        ["fetch", "-t", "dataset", "-l", "2", "-p", f'{{"sort":"{sort_key}"}}', "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.json"))
