"""Traffic-light visualization of appraisal results.

Reads outputs/appraisals.csv and produces the standard Cochrane RoB 2-style
plot: rows are studies, columns are rubric domains + an Overall column.
Each cell is a colored circle with the corresponding symbol:

    Low             green   +
    Some concerns   yellow  -
    High            red     X
    No information  blue    ?

Usage:
    py visualize.py                          # default: rob2 traffic-light
    py visualize.py --rubric robins_i        # plot a different rubric
    py visualize.py --output mychart.png     # custom output path
    py visualize.py --show                   # open plot after saving

Requires matplotlib (`pip install matplotlib`).
"""
import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUTPUTS_DIR = ROOT / "outputs"
APPRAISALS_CSV = OUTPUTS_DIR / "appraisals.csv"

# Cochrane RoB 2 colour convention (also matches the robvis R package).
COLOR = {
    "Low":              "#5cb85c",   # green
    "Some concerns":    "#f0ad4e",   # amber
    "High":             "#d9534f",   # red
    "No information":   "#5bc0de",   # blue
    # ROBINS-I extensions: map to the same palette by severity.
    "Moderate":         "#f0ad4e",
    "Serious":          "#d9534f",
    "Critical":         "#8b0000",   # dark red
}
SYMBOL = {
    "Low":              "+",
    "Some concerns":    "-",
    "High":             "X",
    "No information":   "?",
    "Moderate":         "-",
    "Serious":          "X",
    "Critical":         "X",
}

# Severity order — used to compute the overall rating (worst-of) per study.
SEVERITY = {
    "Low": 0,
    "No information": 1,         # treated as Some concerns for overall purposes
    "Some concerns": 2,
    "Moderate": 2,
    "High": 3,
    "Serious": 3,
    "Critical": 4,
}


def overall_rating(judgments: list[str]) -> str:
    """Return the worst (most severe) judgment across a study's domains."""
    if not judgments:
        return "No information"
    return max(judgments, key=lambda j: SEVERITY.get(j, 1))


def load_rows(path: Path, rubric_id: str) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"No appraisals at {path}. Run `py appraise.py` first.")
    with path.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["rubric"] == rubric_id]
    if not rows:
        with path.open(encoding="utf-8") as f:
            avail = sorted({r["rubric"] for r in csv.DictReader(f)})
        raise SystemExit(f"No appraisals for rubric '{rubric_id}'. Available: {avail}")
    return rows


def build_grid(rows: list[dict]):
    """Pivot CSV rows into (studies, domain_ids, grid[paper][domain] = (judgment, agreement))."""
    studies: "OrderedDict[str, None]" = OrderedDict()
    domains: "OrderedDict[str, str]" = OrderedDict()  # id -> name
    grid: dict[tuple[str, str], tuple[str, str, bool]] = {}
    for r in rows:
        studies.setdefault(r["filename"], None)
        domains.setdefault(r["domain_id"], r["domain_name"])
        grid[(r["filename"], r["domain_id"])] = (
            r["judgment"], r["agreement"], r["flagged"] == "True"
        )
    return list(studies.keys()), list(domains.keys()), domains, grid


def shorten(name: str, n: int = 36) -> str:
    return name if len(name) <= n else name[: n - 3] + "..."


def plot(rows: list[dict], rubric_id: str, out_path: Path, show: bool = False):
    studies, domain_ids, domain_names, grid = build_grid(rows)
    cols = domain_ids + ["Overall"]

    n_studies = len(studies)
    n_cols = len(cols)

    # Geometry
    cell = 0.85          # cell side in inches
    label_w = 3.4        # left-side study-label column width
    legend_h = 1.6       # space below the matrix for legend + domain key

    fig_w = label_w + n_cols * cell + 0.5
    fig_h = 1.0 + n_studies * cell + legend_h
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Coordinate system: each cell centered at integer (col, row).
    ax.set_xlim(-1, n_cols)
    ax.set_ylim(-(n_studies + 1), legend_h + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Header strip background
    ax.add_patch(patches.Rectangle((-0.5, 0.0), n_cols, 0.9,
                                   facecolor="#dcdcdc", edgecolor="black", linewidth=0.5))
    ax.text((n_cols - 1) / 2, 1.3, "Risk of bias domains",
            fontsize=12, fontweight="bold", ha="center")

    # Column headers
    for j, col in enumerate(cols):
        ax.text(j, 0.45, col, fontsize=10, ha="center", va="center", fontweight="bold")

    # Plot each row
    for i, paper in enumerate(studies):
        y = -(i + 1)
        # Study label background (matches reference image style)
        ax.add_patch(patches.Rectangle((-label_w + 0.4, y - 0.45), label_w - 0.4, 0.9,
                                       facecolor="#dcdcdc", edgecolor="black", linewidth=0.5))
        ax.text(-label_w + 0.55, y, shorten(paper), fontsize=8, va="center", ha="left")

        domain_judgments: list[str] = []
        for j, col in enumerate(cols):
            if col == "Overall":
                judgment = overall_rating(domain_judgments)
                agreement = ""
                flagged = False
            else:
                judgment, agreement, flagged = grid.get(
                    (paper, col), ("No information", "0/0", False)
                )
                domain_judgments.append(judgment)

            color = COLOR.get(judgment, "#cccccc")
            sym = SYMBOL.get(judgment, "?")
            edge = "black" if not flagged else "#7a0010"  # darker outline if flagged
            lw = 1.0 if not flagged else 2.5

            circ = patches.Circle((j, y), 0.36, facecolor=color, edgecolor=edge, linewidth=lw)
            ax.add_patch(circ)
            ax.text(j, y, sym, fontsize=14, ha="center", va="center",
                    color="white", fontweight="bold")

    # Vertical separator before Overall column (matches RoB 2 convention).
    sep_x = n_cols - 1.5
    ax.plot([sep_x, sep_x], [-(n_studies + 0.5), 0.9],
            color="black", linewidth=0.8)

    # Domains key (bottom-left)
    key_y = -(n_studies + 1.1)
    ax.text(-label_w + 0.4, key_y, "Domains:", fontsize=9, fontweight="bold", va="top")
    for k, did in enumerate(domain_ids):
        ax.text(-label_w + 0.4, key_y - 0.35 * (k + 1),
                f"{did}: {domain_names[did]}", fontsize=8, va="top")

    # Legend (bottom-right)
    legend_x = n_cols - 1.4
    ax.text(legend_x, key_y, "Judgement", fontsize=9, fontweight="bold", va="top")
    legend_items = [("High", "X"), ("Some concerns", "-"), ("Low", "+"), ("No information", "?")]
    for k, (label, sym) in enumerate(legend_items):
        cy = key_y - 0.35 * (k + 1)
        ax.add_patch(patches.Circle((legend_x, cy), 0.18,
                                     facecolor=COLOR[label], edgecolor="black", linewidth=0.8))
        ax.text(legend_x, cy, sym, fontsize=9, ha="center", va="center",
                color="white", fontweight="bold")
        ax.text(legend_x + 0.35, cy, label, fontsize=8, va="center")

    plt.tight_layout()
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved {out_path} ({n_studies} studies × {len(domain_ids)} domains)")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot the appraisal traffic-light table.")
    parser.add_argument("--rubric", default="rob2",
                        help="Rubric id to plot (default: rob2). Must match a rubric used in appraisals.csv.")
    parser.add_argument("--output", help="Output PNG path (default: outputs/<rubric>_traffic_light.png).")
    parser.add_argument("--show", action="store_true", help="Open the plot window after saving.")
    args = parser.parse_args()

    rows = load_rows(APPRAISALS_CSV, args.rubric)
    out = Path(args.output) if args.output else OUTPUTS_DIR / f"{args.rubric}_traffic_light.png"
    plot(rows, args.rubric, out, show=args.show)


if __name__ == "__main__":
    main()
