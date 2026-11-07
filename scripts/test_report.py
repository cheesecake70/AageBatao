"""Run the test suite and write a human-readable report of every test.

    uv run python scripts/test_report.py            # -> reports/test_report.md
    make test-report

The report lists each test with the first line of its docstring (what it checks), its result and
duration, grouped by file, plus the lint/format status. It is regenerated, never hand-edited.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# One line per test file: what area of the PRD it covers. Shown as the section lead.
FILE_PURPOSE = {
    "test_tokenize.py": "P0 tokeniser and 30k vocabulary (specials, encode/decode, save/load)",
    "test_dedupe.py": "P0 near-duplicate removal (MinHash) in the dataset build",
    "test_filter.py": "P0 sentence filters (length, script, junk) in the dataset build",
    "test_foreign.py": "P0 foreign-language sentence removal in the dataset build",
    "test_naming.py": "Run naming convention e<exp>_<family>_<size>_<data>_<mix>_s<seed>",
    "test_normalize.py": "P0 text normalisation (NFC, URLs, mentions, numbers, emoji)",
    "test_lid.py": "P0 token-level language id used to build the eval buckets",
    "test_dataset.py": "P1 windows: 64-token teacher-forced x/y shift, loaders, eval sentences",
    "test_buckets.py": "P1 §9 bucket assignment: hi / en / switch hi→en / switch en→hi / other",
    "test_metrics.py": "P1 perplexity, top-k rank and <unk> rate per bucket",
    "test_report.py": "P1 §9 markdown/csv result tables",
    "test_ngram.py": "M0 KenLM wrapper: training text, log-prob interface, renormalisation",
    "test_lstm_numpy.py": "M4 NumPy LSTM: shapes, loss, hand-written BPTT vs torch",
    "test_lstm.py": "M1 torch word LSTM: forward, tied weights, parameter counts, log_probs",
    "test_trainer.py": "Trainer: schedule, clipping, early stop, checkpoint and csv log",
    "test_predictor.py": "Demo predictor: partial-word filtering, blocked specials, latency",
    "test_app.py": "FastAPI routes /, /health, POST /predict and input validation",
    "test_summarize.py": "Phase summary: result collection, wide table, acceptance verdict",
    "test_bench.py": "Latency bench: prefix sampling and p50/p95 stats",
}


def docstrings() -> dict[tuple[str, str], str]:
    """(file, test function name) -> first docstring line, parsed from the test sources."""
    out: dict[tuple[str, str], str] = {}
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = ast.get_docstring(node) or ""
                out[(path.name, node.name)] = doc.strip().splitlines()[0] if doc else ""
    return out


def describe(test_name: str) -> str:
    """Fallback when a test has no docstring: 'test_markdown_table_shape' -> 'markdown table shape'."""
    return test_name.removeprefix("test_").replace("_", " ")


def run_pytest(junit: Path) -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"]
    return subprocess.run(cmd, check=False, cwd=ROOT).returncode


def run_lint() -> dict[str, bool]:
    def ok(*cmd: str) -> bool:
        return subprocess.run(cmd, check=False, cwd=ROOT, capture_output=True).returncode == 0

    return {
        "ruff check src tests scripts": ok("ruff", "check", "src", "tests", "scripts"),
        "black --check src tests scripts": ok("black", "--check", "src", "tests", "scripts"),
    }


def parse_junit(junit: Path) -> list[dict]:
    rows = []
    for case in ET.parse(junit).getroot().iter("testcase"):
        file = Path(case.get("file") or case.get("classname", "").replace(".", "/") + ".py").name
        name = case.get("name", "")
        base = name.split("[")[0]
        status = "pass"
        for tag, label in (("failure", "FAIL"), ("error", "ERROR"), ("skipped", "skip")):
            if case.find(tag) is not None:
                status = label
        rows.append(
            {
                "file": file,
                "name": name,
                "base": base,
                "status": status,
                "time": float(case.get("time", 0)),
            }
        )
    return rows


def render(rows: list[dict], lint: dict[str, bool], out: Path) -> None:
    docs = docstrings()
    total = len(rows)
    passed = sum(r["status"] == "pass" for r in rows)
    failed = sum(r["status"] in ("FAIL", "ERROR") for r in rows)
    skipped = sum(r["status"] == "skip" for r in rows)
    seconds = sum(r["time"] for r in rows)
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Test report",
        "",
        f"Generated {ts} by `make test-report`. Do not edit; rerun instead.",
        "",
        "## Summary",
        "",
        "| | |",
        "| --- | --- |",
        f"| tests | {total} |",
        f"| passed | {passed} |",
        f"| failed | {failed} |",
        f"| skipped | {skipped} |",
        f"| wall time | {seconds:.1f} s |",
        f"| python | {sys.version.split()[0]} |",
    ]
    for cmd, good in lint.items():
        lines.append(f"| `{cmd}` | {'clean' if good else 'FAILING'} |")
    verdict = "ALL PASS" if failed == 0 else f"{failed} FAILING"
    lines += ["", f"**Verdict: {verdict}.**", ""]

    by_file: dict[str, list[dict]] = {}
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)

    lines += ["## By file", ""]
    lines += ["| file | tests | passed | time |", "| --- | --- | --- | --- |"]
    for file in sorted(by_file):
        rs = by_file[file]
        lines.append(
            f"| `{file}` | {len(rs)} | {sum(r['status'] == 'pass' for r in rs)} | "
            f"{sum(r['time'] for r in rs):.2f} s |"
        )
    lines.append("")

    lines += ["## Every test", ""]
    for file in sorted(by_file):
        lines += [f"### `{file}`", ""]
        purpose = FILE_PURPOSE.get(file)
        if purpose:
            lines += [purpose, ""]
        lines += ["| test | checks | result | time |", "| --- | --- | --- | --- |"]
        for r in by_file[file]:
            doc = docs.get((file, r["base"]), "") or describe(r["base"])
            mark = {"pass": "pass", "skip": "skip"}.get(r["status"], f"**{r['status']}**")
            lines.append(f"| `{r['name']}` | {doc} | {mark} | {r['time']:.3f} s |")
        lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="reports/test_report.md")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        junit = Path(tmp) / "junit.xml"
        rc = run_pytest(junit)
        rows = parse_junit(junit)
    lint = run_lint()
    out = ROOT / args.out
    render(rows, lint, out)
    failed = sum(r["status"] in ("FAIL", "ERROR") for r in rows)
    print(
        f"{len(rows)} tests, {failed} failing, lint {'clean' if all(lint.values()) else 'FAILING'}"
    )
    print(f"wrote {out.relative_to(ROOT)}")
    sys.exit(0 if rc == 0 and all(lint.values()) else 1)


if __name__ == "__main__":
    main()
