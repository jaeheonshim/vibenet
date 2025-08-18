import csv
import json
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table
from typing_extensions import Annotated

import vibenet
from vibenet import load_model
from vibenet.core import load_audio


class OutputFormat(str, Enum):
    table = "table"
    json = "json"
    csv = "csv"

SR = 16000

app = typer.Typer(no_args_is_help=True)
console = Console()

def _iter_audio_paths(inputs: list[str], recursive: bool, pattern: str|None, quiet: bool):
    paths: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            glob = pattern or "*"
            paths.extend(p.rglob(glob) if recursive else p.glob(glob))
        else:
            if not quiet:
                typer.echo(f"Not found: {inp}", err=True)
            
    return list(sorted(set(paths)))

@app.command()
def predict(
    inputs: Annotated[list[str], typer.Argument(help="Audio file(s) or directory(ies).")],
    recursive: Annotated[bool, typer.Option("--recursive", "-r", help="Recurse into directories.")] = False,
    glob: Annotated[Optional[str], typer.Option("--glob", help='Glob pattern, e.g. "*.mp3"')] = None,
    format: OutputFormat = OutputFormat.table,
    quiet: Annotated[bool, typer.Option("--quiet", "-q")] = False,
    strict: Annotated[bool, typer.Option("--strict", help="Abort on first error.")] = False
):
    net = load_model()
    
    paths = _iter_audio_paths(inputs, recursive, glob, quiet)
    
    rows = []
    
    for p in track(paths, disable=quiet):
        try:
            wf = load_audio(p, SR)
            scores = net.predict(wf, SR)[0]
            row = {'path': str(p), **scores.to_dict()}
            rows.append(row)
        except Exception as e:
            if not quiet:
                typer.echo(f"Failed on {p}: {e}", err=True)
            if strict:
                raise typer.Exit(1)
    
    if format == OutputFormat.table:
        table = Table('path', *vibenet.labels)
        for row in rows:
            table.add_row(row['path'], *['{0:.3f}'.format(row[k]) for k in vibenet.labels])
            
        console.print(table)
    elif format == OutputFormat.csv:
        writer = csv.DictWriter(sys.stdout, ['path', *vibenet.labels])
        writer.writeheader()
        writer.writerows(rows)
    elif format == OutputFormat.json:
        sys.stdout.write(json.dumps(rows))
    
    
if __name__ == '__main__':
    app()