
import hf2dcat.cli as cli

def test_fetch_name_mode_calls_run_fetcher(monkeypatch, tmp_path, runner, app_obj, prog_name):
    called = {}
    def fake_run_fetcher(**kw):
        called.update(kw)
        out = tmp_path / "out.json"
        out.write_text("{}")
        return str(out), {"counts": {"datasets": 1, "models": 0}}

    monkeypatch.setattr(cli, "run_fetcher", lambda **kw: fake_run_fetcher(**kw))

    res = runner.invoke(app_obj, ["fetch", "-d", "nyu-mll/glue", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0
    assert called["dataset_name"] == ["nyu-mll/glue"]
    assert called["model_name"] is None
    assert called["fetch_type"] is None  

def test_fetch_requires_fetch_type_in_batch(monkeypatch, tmp_path, runner, app_obj, prog_name):
    """Test fetching fails without input for name or fetch type"""
    res = runner.invoke(
        app_obj, ["fetch", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code != 0  

def test_fetch_restricted_toggle_passed(monkeypatch, tmp_path, runner, app_obj, prog_name):

    captured = {}
    def fake_run_fetcher(**kw):
        captured.update(kw)
        out = tmp_path / "out.json"
        out.write_text("{}")
        return str(out), {}

    monkeypatch.setattr(cli, "run_fetcher", lambda **kw: fake_run_fetcher(**kw))

    # default: filtered True
    res = runner.invoke(
        app_obj, ["fetch", "-t", "dataset", "-l", "1", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0
    assert captured["filter_restricted"] is True

    # explicit no-filter
    res2 = runner.invoke(
        app_obj, ["fetch", "-t", "dataset", "-l", "1", "-R", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res2.exit_code == 0
    assert captured["filter_restricted"] is False

def test_fetch_dataset_file_missing(runner, app_obj, prog_name):
    """Test fetching datasets fails when the specified input file does not exist"""
    res = runner.invoke(
        app_obj, ["fetch", "-D", "nonexistent.txt", "-o", "some_output"],
        prog_name=prog_name, color=False
    )
    assert res.exit_code != 0
    assert "File not found" in res.stderr or res.stdout

def test_fetch_model_file_missing(runner, app_obj, prog_name):
    """Test fetching models fails when the specified input file does not exist"""
    res = runner.invoke(
        app_obj, ["fetch", "-M", "nonexistent_models.txt", "-o", "outdir"],
        prog_name=prog_name, color=False
    )
    assert res.exit_code != 0
    assert "File not found" in res.stderr or res.stdout