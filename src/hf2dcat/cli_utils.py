import csv
import json
from pathlib import Path
from typing import Optional, List, Union, Dict, Tuple
import click 

# Maximum allowed names from a file
MAX_NAME_LIMIT = 20  

def normalize_formats(formats: List[str]) -> List[str]:
    """Normalize user input to uppercase."""
    return [fmt.upper() for fmt in formats]

def normalize_name_input(name_input: Union[str, List[str]]) -> List[str]:
    """"Normalize a dataset or model name input into a list of non-empty, stripped strings."""
    if isinstance(name_input, str):
        return [name_input.strip()]
    elif isinstance(name_input, list):
        return [str(n).strip() for n in name_input if str(n).strip()]
    else:
        raise TypeError("Expected a string or list of strings")

def load_names_from_file(file_path: Path) -> List[str]:
    """"Load names for datasets or models from a file"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    ext = file_path.suffix.lower()
    names = []

    if ext in {".txt"}:
        names = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif ext == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of names.")
        names = [str(item).strip() for item in data if str(item).strip()]
    elif ext == ".csv":
        with file_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            names = [row[0].strip() for row in reader if row and row[0].strip()]
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .txt, .json, or .csv")

    if not names:
        raise ValueError(f"The file '{file_path.name}' must contain at least one valid name.")

    if len(names) > MAX_NAME_LIMIT:
        raise ValueError(
            f"Too many dataset names provided ({len(names)}). "
            f"Limit is {MAX_NAME_LIMIT} for fetching specific datasets or models by name."
        )
    return names

def prepare_fetch_input(
    dataset_name: Optional[List[str]]= None,
    dataset_name_file: Optional[Path] = None,
    model_name: Optional[List[str]]= None,
    model_name_file: Optional[Path] = None,
    fetch_type: Optional[str] = None,
    limit: Optional[int] = 10, 
    params: Optional[str] = None, 
    output_dir:Optional[Path] = Path("output"),
    filter_restricted: Optional[bool] = True, 
) -> Tuple[str, Optional[List[str]], Optional[List[str]], Optional[str], Optional[int], Optional[Dict], Optional[Path]]:
    """
    Process inputs for dataset and model names (both direct and file-based),
    determine fetch mode, and prepare necessary variables for fetching logic.
    
    Returns:
        (chosen_mode, dataset_name, model_name, fetch_type, limit, params_dict, output_dir)
    """
    dataset_names = []
    model_names = []

    # Handle direct string or list inputs
    if dataset_name:
        dataset_names.extend(normalize_name_input(dataset_name))
    if dataset_name_file:
        try: 
            dataset_names.extend(load_names_from_file(dataset_name_file))  
        except (FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"❌ {e}")

    if model_name:
        model_names.extend(normalize_name_input(model_name))
    if model_name_file:  
        try: 
            model_names.extend(load_names_from_file(model_name_file))
        except (FileNotFoundError, ValueError) as e:
            raise click.ClickException(f"❌ {e}")

    # Deduplicate and sort
    dataset_names = sorted(set(dataset_names)) if dataset_names else None
    model_names = sorted(set(model_names)) if model_names else None

    # Determine fetch mode
    if dataset_names or model_names:
        chosen_mode = "name fetch"
        dataset_name = dataset_names
        model_name = model_names
        fetch_type = None
        limit = None
        params_dict = None
    else:
        chosen_mode = "batch fetch"
        if not fetch_type:
            raise click.ClickException("❌ Error: --fetch-type is required in batch fetch mode.")
        params_dict = json.loads(params) if params else None

    return (chosen_mode, dataset_name, model_name, fetch_type, limit, params_dict, output_dir, filter_restricted)

