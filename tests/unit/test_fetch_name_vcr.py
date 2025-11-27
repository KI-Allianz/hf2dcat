import pytest
from pathlib import Path

@pytest.mark.vcr
def test_fetch_single_dataset_vcr(runner, app_obj, prog_name, tmp_path: Path):
    """Test fetching a single dataset"""
    res = runner.invoke(
        app_obj, ["fetch", "-d", "nyu-mll/glue", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.json"))

@pytest.mark.vcr
def test_fetch_single_model_vcr(runner, app_obj, prog_name, tmp_path: Path):
    """Test fetching a single model"""
    res = runner.invoke(
        app_obj, ["fetch", "-m", "bert-base-uncased", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.json"))

@pytest.mark.vcr
def test_fetch_dataset_names_from_file_vcr(runner, app_obj, prog_name, tmp_path: Path):
    """Test fetching multiple datasets using dataset names from a plain text file."""
    txt = tmp_path / "datasets.txt"
    txt.write_text("nyu-mll/glue\nfacebook/anli\n")
    res = runner.invoke(
        app_obj, ["fetch", "-D", str(txt), "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.json"))

@pytest.mark.vcr
def test_fetch_model_names_from_file_vcr(runner, app_obj, prog_name, tmp_path: Path):
    """Test fetching multiple models using model names from a JSON file"""
    txt = tmp_path / "models.txt"
    txt.write_text("sentence-transformers/all-MiniLM-L6-v2\ngoogle-bert/bert-base-uncased\n")
    res = runner.invoke(
        app_obj, ["fetch", "-M", str(txt), "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.json"))

@pytest.mark.vcr
def test_fetch_dataset_names_from_csv(runner, app_obj, prog_name, tmp_path: Path):
    """Test fetching multiple datasets using dataset names from a CSV file with a header."""
    csv_file = tmp_path / "datasets.csv"
    csv_file.write_text("name\nnyu-mll/glue\nfacebook/anli\n")
    res = runner.invoke(
        app_obj, ["fetch", "-D", str(csv_file), "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.json"))