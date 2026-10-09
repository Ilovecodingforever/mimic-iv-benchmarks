#!/usr/bin/env python
"""Export full and figures-focused summary HTML from one notebook execution."""

from __future__ import annotations

import argparse
import copy
import os
import shutil
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor


TABLE_METADATA_KEY = "summary_table"


def strip_export_cells(nb):
    out = copy.deepcopy(nb)
    kept = []
    for cell in out.cells:
        source = "".join(cell.get("source", ""))
        # ponytail: skip the self-export cell to avoid recursive execution; if
        # more export entrypoints are added, tag them and check the tag here.
        if cell.get("cell_type") == "code" and (
            "export_summary_html.py" in source
            or "nbconvert --to html visualize_summary.ipynb" in source
        ):
            continue
        kept.append(cell)
    out.cells = kept
    return out


def execute_notebook_once(notebook_path):
    nb = nbformat.read(notebook_path, as_version=4)
    nb = strip_export_cells(nb)
    old_mode = os.environ.get("SUMMARY_MODE")
    os.environ["SUMMARY_MODE"] = "full"
    try:
        ExecutePreprocessor(timeout=-1, kernel_name="python3").preprocess(
            nb,
            {"metadata": {"path": str(notebook_path.parent)}},
        )
    finally:
        if old_mode is None:
            os.environ.pop("SUMMARY_MODE", None)
        else:
            os.environ["SUMMARY_MODE"] = old_mode
    return nb


def is_hideable_table_output(output):
    info = output.get("metadata", {}).get(TABLE_METADATA_KEY)
    return isinstance(info, dict) and not info.get("always_show", False)


def figures_notebook(full_nb):
    nb = copy.deepcopy(full_nb)
    nb.cells = [
        cell
        for cell in nb.cells
        if not cell.get("metadata", {}).get("summary_figures_hide", False)
    ]
    for cell in nb.cells:
        if "outputs" in cell:
            cell.outputs = [
                output
                for output in cell.outputs
                if not is_hideable_table_output(output)
            ]
    return nb


def write_html(nb, output_path):
    body, _ = HTMLExporter().from_notebook_node(nb)
    output_path.write_text(body, encoding="utf-8")


def export_summary(notebook_path):
    notebook_path = Path(notebook_path).resolve()
    out_dir = notebook_path.parent
    full_html = out_dir / "visualize_summary_full.html"
    figures_html = out_dir / "visualize_summary_figures.html"
    legacy_html = out_dir / "visualize_summary.html"

    full_nb = execute_notebook_once(notebook_path)
    write_html(full_nb, full_html)
    write_html(figures_notebook(full_nb), figures_html)
    shutil.copyfile(full_html, legacy_html)

    return full_html, figures_html, legacy_html


def self_check():
    nb = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_markdown_cell(
            "table-only heading",
            metadata={"summary_figures_hide": True},
        ),
        nbformat.v4.new_code_cell(outputs=[
            nbformat.v4.new_output(
                "display_data",
                data={"text/html": "<table><tr><td>hide</td></tr></table>"},
                metadata={TABLE_METADATA_KEY: {"always_show": False}},
            ),
            nbformat.v4.new_output(
                "display_data",
                data={"text/html": "<table><tr><td>keep</td></tr></table>"},
                metadata={TABLE_METADATA_KEY: {"always_show": True}},
            ),
            nbformat.v4.new_output(
                "display_data",
                data={"image/png": "abc"},
                metadata={},
            ),
        ]),
    ])
    filtered = figures_notebook(nb)
    assert len(filtered.cells) == 1
    outputs = filtered.cells[0].outputs
    assert len(outputs) == 2
    assert outputs[0].metadata[TABLE_METADATA_KEY]["always_show"] is True
    assert "image/png" in outputs[1].data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebook",
        nargs="?",
        default=Path(__file__).with_name("visualize_summary.ipynb"),
        type=Path,
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        print("self-check passed")
        return
    full_html, figures_html, legacy_html = export_summary(args.notebook)
    print("wrote {}".format(full_html.name))
    print("wrote {}".format(figures_html.name))
    print("copied {} -> {}".format(full_html.name, legacy_html.name))


if __name__ == "__main__":
    main()
