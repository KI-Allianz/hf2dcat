import click
import csv
import json
from pathlib import Path
from typing import Optional, List
from typing_extensions import Annotated
import typer

from huggingface_fetcher import run_fetcher
from huggingface_converter import run_converter
from hf2dcat.cli_utils import normalize_formats, prepare_fetch_input

app = typer.Typer(
        help="Pipeline CLI for fetching and converting Hugging Face metadata to DCAT-AP RDF.", 
        context_settings={"help_option_names": ["-h", "--help"]}
    )

BASE_URI = "https://piveau.io/set/"
VALID_FORMATS = ["rdfxml", "turtle", "jsonld", "ntriples"]
DEFAULT_OUTPUT_FORMATS = ["rdfxml", "turtle"]

# ---------- FETCH COMMAND ----------
@app.command("fetch")
def fetch(
    # --- Name fetch mode ---
    dataset_name: Annotated[
        Optional[List[str]], 
        typer.Option(
            "--dataset-name", "-d",
            help="One or more dataset names to fetch. Use multiple times for multiple names.", 
            rich_help_panel="Name fetch options"
        )
    ] = None,
    model_name: Annotated[
        Optional[List[str]], 
        typer.Option(
            "--model-name", "-m",
            help="One or more model names to fetch. Use multiple times for multiple names.",
            rich_help_panel="Name fetch options"
        )
    ] = None,
    dataset_name_file: Annotated[
        Optional[Path],
        typer.Option(
            "--dataset-name-file", "-D",
            help=("Path to a file containing dataset names."
                  " Supports: .txt (one per line), .json (list of names),"
                  " or .csv (first column, header ignored)."),
            rich_help_panel="Name fetch options"
        )
    ] = None,
    model_name_file: Annotated[
        Optional[Path],
        typer.Option(
            "--model-name-file", "-M",
            help=("Path to a file containing model names."
                  " Supports: .txt (one per line), .json (list of names),"
                  " or .csv (first column, header ignored)."),
            rich_help_panel="Name fetch options"
        )
    ] = None,
    # --- Batch fetch mode ---
    fetch_type: Annotated[
        Optional[str],
        typer.Option(
            "--fetch-type", "-t",
            help="Resource type to fetch in batch mode. Ignored in single fetch mode.",
            click_type=click.Choice(["dataset", "model", "both"], case_sensitive=False),
            rich_help_panel="Batch fetch options"         
        ),       
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option(
            "--limit", "-l", 
            help="Number of items to fetch in batch mode. Ignored in name fetch mode.", 
            rich_help_panel="Batch fetch options"
        ),
    ] = 10,
    params: Annotated[
        Optional[str],
        typer.Option(
            "--params", "-p", 
            help="Optional JSON string for extra params, e.g. '{\"search\": \"text\"}'.",
            rich_help_panel="Batch fetch options"
        ),     
    ] = None,
    filter_restricted: Annotated[
        Optional[bool],
        typer.Option(
            "--filter-restricted/--no-filter-restricted",
            "-r/-R",
            help="Filter out restricted resources (default: enabled). Disable with --no-filter-restricted or -R.", 
            rich_help_panel="Batch fetch options"
        )
    ] = True, 

    # --- General ---
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir", "-o", 
            file_okay=False,
            dir_okay=True, 
            help="Directory where fetched metadata will be saved."
            )
    ] = Path("output"),
):
    """
    Fetch and preprocess metadata for Hugging Face datasets and models.

    Name fetch mode: Fetch metadata for one or more datasets/models using --dataset-name / --model-name or --dataset-name-file / --model-name-file.

    Batch fetch mode: Fetch metadata for top datasets or models by downloads using --fetch-type, with optional --limit, --params and filter_restricted.
    """
    if not any([dataset_name, model_name, dataset_name_file, model_name_file, fetch_type]):
        typer.echo("❌ No input provided. Please provide at least one dataset/model name or use --fetch-type.", err=True)
        raise typer.Exit(code=1)

    chosen_mode, dataset_name, model_name, fetch_type, limit, params_dict, output_dir, filter_restricted= prepare_fetch_input(
        dataset_name, 
        dataset_name_file, 
        model_name, 
        model_name_file, 
        fetch_type, 
        limit, 
        params, 
        output_dir,
        filter_restricted
    )
    # --- Run fetcher ---
    saved_path, _ = run_fetcher(
        fetch_type=fetch_type.lower() if fetch_type else None,
        limit=limit,
        params=params_dict,
        output_dir=output_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        filter_restricted=filter_restricted
    )

    typer.echo(f"\n🤗 Fetch completed ({chosen_mode.upper()} mode). Results saved to: {saved_path}")
  

# ---------- CONVERT COMMAND ----------
@app.command("convert")
def convert(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input-path", "-i",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Path to the input Hugging Face metadata JSON file."
        )
    ],
    output_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--output-dir", "-o",
            file_okay=False,
            dir_okay=True, 
            help="Directory where output files will be saved.",
        )
    ] = Path("output"),
    output_base_name: Annotated[
        Optional[str],
        typer.Option(
            "--output-base-name", "-x",
            help=("Optional base filename (without extension). "
                  "If provided, the resource type (e.g. datasets or models) will be automatically appended "
                  "to distinguish different outputs and avoid overwriting. "
                  "If omitted, a name is auto-generated based on resource type and timestamp.")
        )
    ] = None,
    base_uri: Annotated[
        Optional[str],
        typer.Option(
            "--base-uri", "-b",
            help="Base URI used as namespace for generated resources."
        )
    ] = BASE_URI,
    output_format: Annotated[
        Optional[List[str]],
        typer.Option(
            "--output-format", "-f",
            help="Output formats.",
            click_type=click.Choice(VALID_FORMATS, case_sensitive=False),
        )
    ] = DEFAULT_OUTPUT_FORMATS,
    enable_translation: Annotated[
        Optional[bool],
        typer.Option(
            "--enable-translation/--no-translation",
            "-e/-E",
            help="Enable or disable translation (default: enabled). Disable with --no-translation or -E.",
        )
    ] = True,
):
    """
    Convert Hugging Face metadata to DCAT-AP RDF.
    """
    fmt_list = normalize_formats(output_format)

    created_files = run_converter(
        input_path=input_path,
        output_dir=output_dir,
        output_base=output_base_name,
        base_uri=base_uri,
        enable_translation=enable_translation,
        output_format=fmt_list
    )

    if created_files:
        typer.echo(f"\n🤗 Conversion completed. Created {len(created_files)} RDF files in: {output_dir}")
        # for f in created_files:
        #     typer.echo(f" - {f.name}")
    else:
        typer.echo("❌ Conversion completed but no RDF output files were created.")

@app.command("run-all")
def run_all(
    # ---- Fetcher arguments ----
    dataset_name: Annotated[
        Optional[List[str]], 
        typer.Option(
            "--dataset-name", "-d",
            help="One or more dataset names to fetch. Use multiple times for multiple names.", 
            rich_help_panel="Fetcher options"
        )
    ] = None,
    model_name: Annotated[
        Optional[List[str]], 
        typer.Option(
            "--model-name", "-m",
            help="One or more model names to fetch. Use multiple times for multiple names.", 
            rich_help_panel="Fetcher options"
        )
    ] = None,
    dataset_name_file: Annotated[
        Optional[Path],
        typer.Option(
            "--dataset-name-file", "-D",
            help=("Path to a file containing dataset names."
                  " Supports: .txt (one per line), .json (list of names),"
                  " or .csv (first column, header ignored)."), 
            rich_help_panel="Fetcher options"
        )
    ] = None,
    model_name_file: Annotated[
        Optional[Path],
        typer.Option(
            "--model-name-file", "-M",
        help=("Path to a file containing model names."
              " Supports: .txt (one per line), .json (list of names),"
              " or .csv (first column, header ignored)."), 
            rich_help_panel="Fetcher options"
        )
    ] = None,
    fetch_type: Annotated[
        Optional[str],
        typer.Option(
            "--fetch-type", "-t",
            click_type=click.Choice(["dataset", "model", "both"], case_sensitive=False),
            help="Resource type to fetch in batch mode. Choices: dataset, model, both. Ignored in single fetch mode.",
            rich_help_panel="Fetcher options"
        )
    ] = None,
    limit: Annotated[
        Optional[int],
        typer.Option(
            "--limit", "-l",
            help="Number of items to fetch in batch mode. Ignored in single fetch mode.",
            rich_help_panel="Fetcher options"
        )
    ] = 10,
    params: Annotated[
        Optional[str],
        typer.Option(
            "--params", "-p",
            help='Optional JSON string for extra params, e.g. \'{"search": "text"}\'.',
            rich_help_panel="Fetcher options"
        )
    ] = None,
    filter_restricted: Annotated[
        Optional[bool],
        typer.Option(
            "--filter-restricted/--no-filter-restricted",
            "-r/-R",
            help="Filter out restricted resources (default: enabled). Disable with --no-filter-restricted or -R.",
            rich_help_panel="Fetcher options"
        )
    ] = True, 
    # ---- Shared output dir ----
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o",
            help="Directory where fetched metadata and converted RDF files will be saved.",
            rich_help_panel="Shared options"
        )
    ] = Path("output"),

    # ---- Converter arguments ----
    output_base_name: Annotated[
        Optional[str],
        typer.Option(
            "--output-base-name", "-x",
            help=("Optional base filename (without extension) for converted RDF files. "
                  "If provided, the resource type (e.g. datasets or models) will be automatically appended "
                  "to distinguish different outputs and avoid overwriting. "
                  "If omitted, a name is auto-generated based on resource type and timestamp."), 
            rich_help_panel="Converter options"
        )
    ] = None,
    base_uri: Annotated[
        Optional[str],
        typer.Option(
            "--base-uri", "-b",
            help="Base URI used as namespace for generated resources.",
            rich_help_panel="Converter options"
        )
    ] = BASE_URI,
    output_format: Annotated[
        Optional[List[str]],
        typer.Option(
            "--output-format", "-f",
            click_type=click.Choice(VALID_FORMATS, case_sensitive=False),
            help="Output formats. Choices: RDFXML, TURTLE, JSONLD, NTRIPLES. (default: RDFXML and TURTLE).",
            rich_help_panel="Converter options"
        )
    ] = DEFAULT_OUTPUT_FORMATS,
    enable_translation: Annotated[
        Optional[bool],
        typer.Option(
            "--enable-translation/--no-translation", "-e/-E",
            help="Enable or disable translation (default: enabled). Disable with --no-translation or -E.",
            rich_help_panel="Converter options"
        )
    ] = True,
):
    """
    Run the complete pipeline: fetch metadata from Hugging Face and convert it to DCAT-AP RDF.
    """

    if not any([dataset_name, model_name, dataset_name_file, model_name_file, fetch_type]):
        typer.echo("❌ No input provided. Please provide at least one dataset/model name or use --fetch-type to start the fetching step.", err=True)
        raise typer.Exit(code=1)

    chosen_mode, dataset_name, model_name, fetch_type, limit, params_dict, output_dir, filter_restricted = prepare_fetch_input(
        dataset_name, 
        dataset_name_file, 
        model_name, 
        model_name_file, 
        fetch_type, 
        limit, 
        params, 
        output_dir,
        filter_restricted
    )

    # --- Run fetcher ---
    typer.echo("\nStep 1 of 2: Fetching metadata from Hugging Face...\n")
    json_path, _ = run_fetcher(
        fetch_type=fetch_type.lower() if fetch_type else None,
        limit=limit,
        params=params_dict,
        output_dir=output_dir,
        dataset_name=dataset_name,
        model_name=model_name,
        filter_restricted=filter_restricted
    )
    # typer.echo(f"\n🤗 Fetched metadata saved to: {json_path}")

    # --- Run converter ---
    typer.echo("\nStep 2 of 2: Converting fetched metadata to RDF...\n")
    fmt_list = normalize_formats(output_format)
    created_files = run_converter(
        input_path=json_path,
        output_dir=output_dir,
        output_base=output_base_name,
        base_uri=base_uri,
        enable_translation=enable_translation,
        output_format=fmt_list
    )

    if created_files:
        typer.echo(f"\n🤗 Pipeline Completed. Created {len(created_files)} RDF files in: {output_dir}")
        # for f in created_files:
        #     typer.echo(f" - {f.name}")
    else:
        typer.echo("❌ Pipeline completed but no RDF output files were created.")

    
if __name__ == "__main__":
    app()
