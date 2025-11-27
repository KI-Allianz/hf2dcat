import pytest
from pathlib import Path
from hf2dcat.cli import run_fetcher, run_converter  

@pytest.mark.vcr
def test_run_fetcher_then_converter_vcr(tmp_path: Path):
    
    # Step 1: Fetch 
    saved_path, results = run_fetcher(
        fetch_type=None,
        limit=None,
        params=None,
        output_dir=tmp_path,
        dataset_name=["nebius/SWE-rebench"],
        model_name=None,
        filter_restricted=True,
    )
    assert Path(saved_path).exists()
    assert results["counts"]["total"] >= 1

    # Step 2: Convert with translation enabled
    created_files = run_converter(
        input_path=Path(saved_path),
        output_dir=tmp_path,
        output_base=None,
        enable_translation=True,
    )
    assert any(p.suffix == ".ttl" for p in created_files)
    assert any(p.suffix == ".rdf" for p in created_files)

@pytest.mark.vcr
def test_converter_no_translation_vcr(tmp_path: Path):
    # Step 1: Fetch 
    saved_path, results = run_fetcher(
        fetch_type=None,
        limit=None,
        params=None,
        output_dir=tmp_path,
        dataset_name=["nebius/SWE-rebench"],
        model_name=None,
        filter_restricted=True,
    )
    assert Path(saved_path).exists()
    assert results["counts"]["total"] >= 1

    # Step 2: Convert with no translation
    created_files = run_converter(
        input_path=Path(saved_path),
        output_dir=tmp_path,
        output_base=None,
        enable_translation=False,
    )
    assert any(p.suffix == ".ttl" for p in created_files)
    assert any(p.suffix == ".rdf" for p in created_files)

@pytest.mark.vcr
def test_convert_model(runner, app_obj, prog_name, tmp_path: Path):
    # Step 1: fetch HF model metadata
    fetch = runner.invoke(
        app_obj,
        ["fetch", "-m", "sentence-transformers/all-MiniLM-L6-v2", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert fetch.exit_code == 0, fetch.stdout
    
    input_json = max(tmp_path.glob("*.json"), key=lambda p: p.stat().st_mtime)  

    # Step 2: convert JSON metadata to RDF
    res = runner.invoke(
        app_obj, ["convert", "-i", str(input_json), "-o", str(tmp_path), "-f", "turtle"],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.ttl"))

@pytest.mark.vcr
def test_convert_model(runner, app_obj, prog_name, tmp_path: Path):
    # Step 1: fetch HF dataset metadata
    fetch = runner.invoke(
        app_obj,
        ["fetch", "-d", "nebius/SWE-rebench", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert fetch.exit_code == 0, fetch.stdout
    
    input_json = max(tmp_path.glob("*.json"), key=lambda p: p.stat().st_mtime)  

    # Step 2: convert JSON metadata to RDF
    res = runner.invoke(
        app_obj, ["convert", "-i", str(input_json), "-o", str(tmp_path), "-f", "turtle"],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.ttl"))

@pytest.mark.vcr
@pytest.mark.parametrize("ftype", ["dataset", "model", "both"])
def test_convert_model(runner, app_obj, prog_name, ftype, tmp_path: Path):
    # Step 1: batch fetch HF metadata
    fetch = runner.invoke(
        app_obj,
        ["fetch", "-t", ftype, "-l", "2", "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert fetch.exit_code == 0, fetch.stdout
    
    input_json = max(tmp_path.glob("*.json"), key=lambda p: p.stat().st_mtime)  

    # Step 2: convert JSON metadata to RDF
    res = runner.invoke(
        app_obj, ["convert", "-i", str(input_json), "-o", str(tmp_path)],
        prog_name=prog_name, color=False
    )
    assert res.exit_code == 0, res.stdout
    assert list(tmp_path.glob("*.ttl"))

