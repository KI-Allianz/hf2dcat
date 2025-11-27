import pytest
from pathlib import Path

@pytest.mark.vcr
@pytest.mark.parametrize("ftype", ["dataset", "model", "both"])
def test_fetch_batch_vcr(runner, app_obj, prog_name, tmp_path: Path, ftype):
    res = runner.invoke(
        app_obj, ["fetch", "-t", ftype, "-l", "2", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.json"))
