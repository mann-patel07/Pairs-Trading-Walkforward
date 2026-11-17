"""Check that the README numbers match results/ and that every named path exists.

Usage: python check_consistency.py
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
IGNORED_DIRS = {".venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".claude"}
IGNORED_FILES = {".DS_Store", "NOTES_FOR_MANN.md"}


def pct(x: float) -> str:
    """Format a fraction as a percent with two decimals."""
    return f"{x * 100:.2f}%"


def main() -> int:
    """Run all checks and return a process exit code."""
    readme = (ROOT / "README.md").read_text()
    summary = pd.read_csv(ROOT / "results" / "summary.csv")
    boot = pd.read_csv(ROOT / "results" / "bootstrap.csv").set_index("quantity")
    net = summary[summary.basis == "net"].set_index("variant")
    failures = []
    checked = 0

    def expect(label: str, expected: str, found: str) -> None:
        nonlocal checked
        checked += 1
        if expected != found:
            failures.append(f"{label}: README has {found!r}, results give {expected!r}")

    p = net.loc["COINT"]
    sh = boot.loc["net_sharpe_COINT"]
    m = re.search(r"Net annualized return: (\S+)", readme)
    expect("headline return", pct(p.ann_return), m.group(1) if m else "missing")
    m = re.search(r"Net annualized Sharpe ratio: (\S+) .*?: (\S+) to (\S+)\)", readme)
    found = m.groups() if m else ("missing",) * 3
    expect("headline Sharpe", f"{p.sharpe:.2f}", found[0])
    expect("Sharpe interval low", f"{sh.ci_low:.2f}", found[1])
    expect("Sharpe interval high", f"{sh.ci_high:.2f}", found[2])
    expect("bootstrap point estimate", f"{p.sharpe:.2f}", f"{sh.estimate:.2f}")
    m = re.search(r"Max drawdown: (\S+)", readme)
    expect("headline drawdown", pct(p.max_drawdown), m.group(1) if m else "missing")

    for variant in ["ALL", "COINT", "COINT_BH"]:
        row = re.search(rf"^\| {variant} \|(.*)\|$", readme, flags=re.M)
        cells = [c.strip() for c in row.group(1).split("|")] if row else ["missing"] * 6
        r = net.loc[variant]
        expected = [pct(r.ann_return), pct(r.ann_volatility), f"{r.sharpe:.2f}",
                    pct(r.max_drawdown), str(int(r.pair_windows_traded)), str(int(r.completed_trades))]
        for name, exp, got in zip(["return", "volatility", "Sharpe", "drawdown", "pair-windows", "trades"],
                                  expected, cells):
            expect(f"table {variant} {name}", exp, got)

    named = set(re.findall(r"\(((?:Images)/[^)]+)\)", readme))
    layout = readme.split("## File layout")[1].split("## How to run")[0]
    named |= set(re.findall(r"^- `([^`]+)`", layout, flags=re.M))
    for path in sorted(named):
        checked += 1
        if not (ROOT / path).exists():
            failures.append(f"README names a path that does not exist: {path}")
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if path.is_dir() or ".git" in rel.parts or set(rel.parts) & IGNORED_DIRS or path.name in IGNORED_FILES:
            continue
        checked += 1
        if str(rel) not in named:
            failures.append(f"File exists but is not listed in the README layout: {rel}")

    if failures:
        print("CONSISTENCY CHECK FAILED")
        for f in failures:
            print(" -", f)
        return 1
    print(f"Consistency check passed: {checked} checks (headline, table, paths, layout).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
