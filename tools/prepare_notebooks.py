#!/usr/bin/env python3
"""Create a sanitized, non-executed archive of the original thesis notebooks.

Usage:
    python tools/prepare_notebooks.py --source-dir /path/to/only-codes \
        --output-dir notebooks

Add --validate to check the generated documents with the optional nbformat
package. Code cells are AST-checked, never executed. Cells containing IPython
magics or shell commands are recorded and skipped by that syntax check.

Only notebook source is retained. Training outputs, execution counts, embedded
attachments and arbitrary metadata are removed. Windows absolute paths become
explicit /path/to/private-data placeholders; individual image filenames become
example.<extension>. Relative model names and upstream attribution remain.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re


EXPERIMENTS = {
    "GAN_anomaly detection.ipynb",
    "ganomaly_ten.ipynb",
    "tensorflow_webgan.ipynb",
    "pytorchanomalib_GANomaly.ipynb",
    "PYtorch.ipynb",
    "patchCNN_test.ipynb",
    "exception.ipynb",
    "Pretrained_featrue_extraction.ipynb",
    "Pretrained_model_autoencoder.ipynb",
}
# The opening quote is required, so https:// upstream links cannot match a drive.
WINDOWS_LITERAL = re.compile(
    r"(?P<prefix>[rRuUbBfF]{0,2})(?P<quote>['\"])(?P<path>[A-Za-z]:[\\/][^'\"\r\n]*)(?P=quote)"
)
IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif"}
ANY_WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\+|/(?!/))")
KERNEL_METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def classify(filename):
    if filename in EXPERIMENTS:
        if filename == "ganomaly_ten.ipynb":
            return "experiments", "external_launcher"
        if filename == "pytorchanomalib_GANomaly.ipynb":
            return "experiments", "experimental_separate_stack"
        return "experiments", "unfinished_prototype"
    if "vae" in filename.lower() or "variational" in filename.lower():
        return "variational_autoencoders", "historical_research"
    return "autoencoders", "historical_research"


def private_path(original):
    """Keep useful dataset/class folders without machine/user identifiers."""
    normalized = re.sub(r"\\+", "/", original)
    parts = [part for part in normalized.split("/") if part]
    if len(parts) > 3 and parts[1].lower() == "users":
        parts = parts[3:]  # drive, Users, account
        if parts and parts[0].lower() in {"desktop", "documents", "downloads"}:
            parts = parts[1:]
        else:
            # An arbitrary personal environment name is not useful provenance.
            parts = parts[-1:]
    else:
        parts = parts[1:]
    if parts and Path(parts[-1]).suffix.lower() in IMAGE_EXTENSIONS:
        parts[-1] = "example" + Path(parts[-1]).suffix.lower()
    return "/path/to/private-data/" + "/".join(parts)


def sanitize_source(source):
    if isinstance(source, list):
        source = "".join(source)
    if not isinstance(source, str):
        raise ValueError("Cell source must be a string or list of strings")

    def replace_path(match):
        return (
            match.group("prefix")
            + match.group("quote")
            + private_path(match.group("path"))
            + match.group("quote")
        )

    source, count = WINDOWS_LITERAL.subn(replace_path, source)
    source = re.sub(
        r"!\[[^\]]*\]\((?:attachment:|data:image/)[^)]*\)",
        "[Image omitted from historical notebook.]",
        source,
        flags=re.IGNORECASE,
    )
    return source, count


def check_code(cells):
    parsed = 0
    skipped = []
    errors = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(line.lstrip().startswith(("%", "!", "?")) for line in source.splitlines()):
            skipped.append(index)
            continue
        try:
            ast.parse(source)
            parsed += 1
        except SyntaxError as exc:
            errors.append({"cell": index, "line": exc.lineno, "message": exc.msg})
    return {"ast_parsed_cells": parsed, "ipython_cells_skipped": skipped, "syntax_errors": errors}


def sanitize_notebook(document, filename):
    cells = []
    redacted_paths = 0
    for index, original in enumerate(document.get("cells", [])):
        kind = original.get("cell_type")
        if kind not in {"code", "markdown", "raw"}:
            raise ValueError(f"{filename}: unsupported cell type {kind!r}")
        source, count = sanitize_source(original.get("source", []))
        redacted_paths += count
        # IDs are stable and contain no source-machine metadata.
        cell_id = "cell-" + sha256(f"{filename}:{index}".encode("utf-8"))[:12]
        cell = {
            "cell_type": kind,
            "id": cell_id,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        cells.append(cell)
    notebook = {
        "cells": cells,
        "metadata": KERNEL_METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook, redacted_paths, check_code(cells)


def prepare(source_dir, output_dir, validate=False):
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("Output must be a separate archive directory, not the source directory")
    paths = sorted(source_dir.glob("*.ipynb"), key=lambda path: path.name.casefold())
    if not paths:
        raise ValueError("Source directory contains no .ipynb files")
    nbformat = None
    if validate:
        import nbformat as notebook_format

        nbformat = notebook_format
    manifest = {
        "format_version": 1,
        "generator": "tools/prepare_notebooks.py",
        "scope": "Historical, sanitized notebook source; no notebook code was executed.",
        "sanitization": [
            "Remove all saved outputs and execution counts.",
            "Remove attachments and arbitrary notebook/cell metadata; use a Python 3 kernel.",
            "Replace absolute Windows paths and individual private-image filenames with placeholders.",
            "Preserve notebook filenames, source logic, relative model names and upstream attribution.",
        ],
        "notebooks": [],
    }
    for path in paths:
        raw = path.read_bytes()
        notebook, redactions, validation = sanitize_notebook(json.loads(raw), path.name)
        if nbformat is not None:
            nbformat.validate(nbformat.from_dict(notebook))
        category, status = classify(path.name)
        destination = output_dir / category / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
        destination.write_bytes(content)
        manifest["notebooks"].append(
            {
                "filename": path.name,
                "path": destination.relative_to(output_dir).as_posix(),
                "category": category,
                "status": status,
                "original_sha256": sha256(raw),
                "sanitized_sha256": sha256(content),
                "paths_redacted": redactions,
                "code_validation": validation,
            }
        )
    manifest["count"] = len(manifest["notebooks"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def check_archive(archive_dir):
    """Read and verify a prepared archive without modifying or executing it."""
    import nbformat

    archive_dir = archive_dir.resolve()
    manifest = json.loads((archive_dir / "source-manifest.json").read_text(encoding="utf-8"))
    entries = manifest["notebooks"]
    if manifest.get("count") != len(entries):
        raise ValueError("Manifest count does not match notebook entries")
    expected = set()
    total_bytes = 0
    for entry in entries:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".ipynb":
            raise ValueError("Manifest paths must stay within the notebook archive")
        if entry["path"] in expected:
            raise ValueError("Manifest contains a duplicate notebook path")
        expected.add(entry["path"])
        if relative.name != entry["filename"]:
            raise ValueError("Manifest filename differs from the notebook path")
        category, status = classify(relative.name)
        if entry["category"] != category or entry["status"] != status or relative.parts[0] != category:
            raise ValueError(f"Unexpected category/status: {relative.name}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["original_sha256"]):
            raise ValueError("Original-source SHA256 has an invalid format")
        raw = (archive_dir / relative).read_bytes()
        total_bytes += len(raw)
        if sha256(raw) != entry["sanitized_sha256"]:
            raise ValueError(f"Sanitized SHA256 mismatch: {relative}")
        document = json.loads(raw)
        nbformat.validate(document)
        if document.get("metadata") != KERNEL_METADATA:
            raise ValueError(f"Unexpected notebook metadata: {relative}")
        for cell in document["cells"]:
            if cell.get("metadata") != {} or "attachments" in cell:
                raise ValueError(f"Cell metadata or attachments remain: {relative}")
            if cell["cell_type"] == "code":
                if cell.get("outputs") != [] or cell.get("execution_count") is not None:
                    raise ValueError(f"Saved execution state remains: {relative}")
            source = "".join(cell["source"])
            if ANY_WINDOWS_PATH.search(source) or "data:image/" in source.lower():
                raise ValueError(f"Private absolute path or embedded image remains: {relative}")
        validation = check_code(document["cells"])
        if validation != entry["code_validation"] or validation["syntax_errors"]:
            raise ValueError(f"Code validation differs or contains syntax errors: {relative}")
    actual = {path.relative_to(archive_dir).as_posix() for path in archive_dir.rglob("*.ipynb")}
    if actual != expected:
        raise ValueError("Notebook files differ from the manifest inventory")
    return {"count": len(entries), "notebook_bytes": total_bytes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-dir", type=Path)
    mode.add_argument("--check", type=Path, metavar="ARCHIVE_DIR", help="Verify an existing archive; requires nbformat")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "notebooks")
    parser.add_argument("--validate", action="store_true", help="Validate documents with optional nbformat")
    args = parser.parse_args()
    if args.check is not None:
        result = check_archive(args.check)
        print(f"Verified {result['count']} sanitized notebooks ({result['notebook_bytes']:,} bytes); no cells executed.")
        return
    manifest = prepare(args.source_dir, args.output_dir, args.validate)
    entries = manifest["notebooks"]
    print(f"Prepared {len(entries)} notebooks; no cells executed.")
    print(f"Redacted {sum(entry['paths_redacted'] for entry in entries)} absolute Windows paths.")
    print(f"AST syntax errors: {sum(len(entry['code_validation']['syntax_errors']) for entry in entries)}")
    print(f"IPython cells recorded, not executed: {sum(len(entry['code_validation']['ipython_cells_skipped']) for entry in entries)}")


if __name__ == "__main__":
    main()
