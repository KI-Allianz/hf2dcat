import pytest
import re

@pytest.fixture(autouse=True)
def wide_terminal(monkeypatch):
    monkeypatch.setenv("COLUMNS", "120")
    yield

def _normalize_help(s: str) -> str:
    s = re.sub(r"\x1b\[[0-9;]*m", "", s)                      
    s = re.sub(r"[^\x00-\x7F]+", " ", s)    
    s = re.sub(r"\s+", " ", s).strip()     
    return s

def test_root_help(runner, app_obj, prog_name):
    """Test overall help"""
    res = runner.invoke(app_obj, ["--help"], prog_name=prog_name, color=False)
    assert res.exit_code == 0
    norm_stdout = _normalize_help(res.stdout)
    assert f"Usage: {prog_name}" in norm_stdout
    assert "Commands" in norm_stdout
    for cmd in ["fetch", "convert", "run-all"]:
        assert cmd in norm_stdout

@pytest.mark.parametrize(
    "cmd, contained_text",
    [
        ("fetch", [
            "Fetch and preprocess metadata", 
            "Name fetch mode", 
            "Batch fetch mode", 
            "--dataset-name", 
            "-d", 
            "--model-name", 
            "-m", 
            "--fetch-type",
            "-t", 
            "--limit",
            "-l",
            "--params",
            "-p"
        ]),
        ("convert", [
            "Convert Hugging Face metadata",
            "--input-path",
            "-i",
            "--output-dir",
            "-o"
        ]),
        ("run-all", [
            "Run the complete pipeline",
            "--dataset-name", 
            "-d", 
            "--model-name", 
            "-m", 
            "--fetch-type",
            "-t", 
            "--limit",
            "-l",
            "--params",
            "-p",
            "--output-dir",
            "-o",
            "--output-format",
            "-f"
        ]),
    ],
)
def test_subcommand_help(runner, app_obj, prog_name, cmd, contained_text):
    """Test each subcommand help"""
    res = runner.invoke(app_obj, [cmd, "--help"], prog_name=prog_name)
    assert res.exit_code == 0
    norm_stdout = _normalize_help(res.stdout)
    assert f"Usage: {prog_name} {cmd}" in norm_stdout
    for text in contained_text:
        assert text in norm_stdout


@pytest.mark.parametrize("sub", [None, "fetch", "convert", "run-all"])
def test_short_help(runner, app_obj, prog_name, sub):
    """Test subcommand short help"""
    args = ["-h"] if sub is None else [sub, "-h"]
    res = runner.invoke(app_obj, args, prog_name=prog_name)
    assert res.exit_code == 0
    assert f"Usage: {prog_name}" in res.stdout

def test_invalid_subcommand(runner, app_obj, prog_name):
    """Test handling of invalid subcommands"""
    res = runner.invoke(app_obj, ["invalid-command"], prog_name=prog_name)
    assert res.exit_code != 0
    norm_stdout = _normalize_help(res.stdout or res.stderr)
    assert "No such command" in norm_stdout

@pytest.mark.parametrize("sub", ["fetch", "convert", "run-all"])
def test_missing_required_args(runner, app_obj, prog_name, sub):
    """Test subcommands without required arguments"""
    res = runner.invoke(app_obj, sub, prog_name=prog_name)
    assert res.exit_code != 0
    norm_stdout = _normalize_help(res.stdout or res.stderr)
    assert "No input provided" in norm_stdout or "Usage" in norm_stdout