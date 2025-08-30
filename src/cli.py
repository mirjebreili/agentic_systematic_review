"""
The Command-Line Interface (CLI) for the LocalAgent-SR application.
It provides commands to run the systematic review pipeline.
"""
import logging
from pathlib import Path
import typer
from typing_extensions import Annotated

# --- Constants for Template Files ---

INCLUSION_YAML_TEMPLATE = """\
version: 1.0
logic:
  # All criteria in this list must be met for a paper to be included.
  require_all:
    - id: is_rct
      description: "The study must be a Randomized Controlled Trial (RCT). Look for explicit mentions of 'randomized', 'randomly assigned', etc."
      evidence_must_be_quoted: true
    - id: is_human_subjects
      description: "The study participants must be humans."
      evidence_must_be_quoted: true

  # If any criterion in this list is met, the paper is automatically excluded.
  exclude_if_any:
    - id: is_review_article
      description: "The paper is a meta-analysis, systematic review, or literature review, not a primary study."

notes: "If conflicting evidence is found for any criterion, the agent should mark the paper's status as UNSURE and flag it for manual review."
"""

FEATURES_YAML_TEMPLATE = """\
version: 1.0
features:
  - name: sample_size
    type: integer
    description: "Extract the total number of participants analyzed in the study. Prioritize numbers labeled 'analyzed' or 'per-protocol' over 'enrolled' or 'recruited'."
    regex_hints: ["\\\\bN\\\\s*=\\\\s*(\\\\d+)", "sample of (\\\\d+)"]

  - name: intervention_group_1_details
    type: string
    description: "Describe the primary intervention administered to the main experimental group. E.g., '10mg of Drug X daily for 4 weeks'."

  - name: primary_outcome_measure
    type: string
    description: "Identify the name of the primary outcome measure used to evaluate the intervention's effect. E.g., 'Hamilton Depression Rating Scale (HDRS) score'."
"""

# --- CLI Application ---

app = typer.Typer(help="LocalAgent-SR: A multi-agent systematic review tool.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _check_and_create_config(root_path: Path, filename: str, template: str):
    """Checks for a config file and offers to create it from a template if missing."""
    config_file = root_path / filename
    if not config_file.exists():
        logging.warning(f"Configuration file '{filename}' not found in '{root_path}'.")
        create_file = typer.confirm(f"Do you want to create a default '{filename}'?")
        if create_file:
            with open(config_file, "w", encoding="utf-8") as f:
                f.write(template)
            logging.info(f"'{filename}' created successfully.")
            typer.echo(f"A default '{filename}' has been created at '{config_file}'. Please review and edit it to fit your needs.")
        else:
            logging.error(f"Aborting: '{filename}' is required to proceed.")
            raise typer.Exit(code=1)

@app.command()
def run_all(
    root_path: Annotated[Path, typer.Option("--root", help="Path to the directory containing papers and configuration.", exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True)],
    model: Annotated[str, typer.Option("--model", help="Name of the Ollama model to use for analysis.")] = "llama3",
):
    """
    Runs the complete systematic review pipeline: Ingest, Screen, Extract, and Export.
    """
    logging.info(f"Starting complete run with root='{root_path}' and model='{model}'")

    # 1. Check for configuration files
    _check_and_create_config(root_path, "inclusion.yaml", INCLUSION_YAML_TEMPLATE)
    _check_and_create_config(root_path, "features.yaml", FEATURES_YAML_TEMPLATE)

    # 2. Placeholder for pipeline execution
    typer.echo("\n" + "="*20)
    typer.echo("Pipeline Execution (Placeholder)")
    typer.echo(f"  - Root Path: {root_path}")
    typer.echo(f"  - Ollama Model: {model}")
    typer.echo("  - TODO: Initialize and run the LangGraph orchestrator.")
    typer.echo("="*20 + "\n")

    # In the future, this will involve:
    # - Creating a timestamped output directory.
    # - Initializing the AppState.
    # - Invoking the LangGraph manager.
    # - Printing a final summary.

    logging.info("Placeholder run finished successfully.")
    typer.secho("✅ Pipeline finished (placeholder).", fg=typer.colors.GREEN)

# --- Other Commands (Placeholders) ---

@app.command(short_help="Ingest and parse documents.")
def ingest():
    """(Placeholder) Ingests and parses documents from the source directory."""
    typer.echo("This command will one day handle just the ingestion and parsing step.")

@app.command(short_help="Screen documents for inclusion.")
def screen():
    """(Placeholder) Runs the screening agent on ingested documents."""
    typer.echo("This command will one day run only the screening step.")

@app.command(short_help="Extract features from included documents.")
def extract():
    """(Placeholder) Runs the extraction agent on included documents."""
    typer.echo("This command will one day run only the feature extraction step.")

if __name__ == "__main__":
    app()
