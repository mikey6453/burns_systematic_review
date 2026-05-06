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
    # Newcastle-Ottawa star ratings (higher is better — opposite polarity).
    "0 stars":          "#d9534f",   # red — worst
    "1 star":           "#5cb85c",   # green — got the star
    "2 stars":          "#3d8b3d",   # darker green — got both stars (Comparability)
    # JBI Case-series checklist (Yes is good, No is bad).
    "Yes":              "#5cb85c",
    "No":               "#d9534f",
    "Unclear":          "#f0ad4e",
    "Not applicable":   "#cccccc",
}
SYMBOL = {
    "Low":              "+",
    "Some concerns":    "-",
    "High":             "X",
    "No information":   "?",
    "Moderate":         "-",
    "Serious":          "X",
    "Critical":         "X",
    "0 stars":          "0",
    "1 star":           "1",
    "2 stars":          "2",
    "Yes":              "Y",
    "No":               "N",
    "Unclear":          "?",
    "Not applicable":   "-",
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
    # NOS: star count is the inverse of severity (more stars = better study).
    "2 stars": 0,
    "1 star":  0,
    "0 stars": 3,
    # Case-series: Yes is good (low severity), No/Unclear are bad.
    "Yes":              0,
    "No":               3,
    "Unclear":          1,
    "Not applicable":   0,
}


def _short_name(name: str) -> str:
    """Trim redundant section prefixes like 'Selection: ' from NOS domain names."""
    parts = name.split(":", 1)
    return parts[1].strip() if len(parts) == 2 and len(parts[1]) > 4 else name


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


def shorten(name: str, n: int = 24) -> str:
    return name if len(name) <= n else name[: n - 3] + "..."


def _legend_for_rubric(rubric_id: str) -> list[tuple[str, str]]:
    """Pick the legend entries appropriate for the rubric's judgment scale."""
    if rubric_id == "nos_cohort":
        return [("2 stars", "2"), ("1 star", "1"), ("0 stars", "0")]
    if rubric_id == "robins_i":
        return [("Critical", "X"), ("Serious", "X"), ("Moderate", "-"),
                ("Low", "+"), ("No information", "?")]
    if rubric_id == "case_series":
        return [("Yes", "Y"), ("No", "N"), ("Unclear", "?"), ("Not applicable", "-")]
    # rob2 / custom default
    return [("High", "X"), ("Some concerns", "-"), ("Low", "+"), ("No information", "?")]


def plot(rows: list[dict], rubric_id: str, out_path: Path | None = None,
         show: bool = False, return_fig: bool = False):
    studies, domain_ids, domain_names, grid = build_grid(rows)

    n_studies = len(studies)
    n_dom = len(domain_ids)
    legend_items = _legend_for_rubric(rubric_id)

    # Geometry
    cell = 0.85          # cell side in inches
    label_w = 4.6        # widened so long filenames don't bleed into cells
    overall_gap = 0.7    # extra horizontal gap before the Overall column
    # Column x-positions: domain columns at 0..n_dom-1, Overall to the right with a gap.
    domain_xs = list(range(n_dom))
    overall_x = n_dom - 1 + overall_gap + 1   # last domain index + gap + 1 cell
    col_xs = domain_xs + [overall_x]
    cols = domain_ids + ["Overall"]

    # Bottom area: domain key (one row per domain) THEN legend (one row per item).
    line_h = 0.4
    key_lines = 1 + n_dom
    legend_lines = 1 + len(legend_items)
    bottom_h = (key_lines + legend_lines + 2) * line_h

    fig_w = label_w + (overall_x + 1.0) * cell + 0.5
    fig_h = max(4.5, 1.0 + n_studies * cell + bottom_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.set_xlim(-label_w / cell, overall_x + 1.0)
    ax.set_ylim(-(n_studies + key_lines + legend_lines + 3), 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Header strip background — extends across all columns including the Overall gap.
    header_w = overall_x + 1.0
    ax.add_patch(patches.Rectangle((-0.5, 0.0), header_w, 0.9,
                                   facecolor="#dcdcdc", edgecolor="black", linewidth=0.5))
    ax.text((header_w - 1) / 2, 1.3, "Risk of bias domains",
            fontsize=12, fontweight="bold", ha="center")

    # Column headers
    for j, col in enumerate(cols):
        ax.text(col_xs[j], 0.45, col, fontsize=10, ha="center", va="center", fontweight="bold")

    # Plot each row
    for i, paper in enumerate(studies):
        y = -(i + 1)
        ax.add_patch(patches.Rectangle((-label_w + 0.4, y - 0.45), label_w - 0.4, 0.9,
                                       facecolor="#dcdcdc", edgecolor="black", linewidth=0.5))
        ax.text(-label_w + 0.55, y, shorten(paper), fontsize=8, va="center", ha="left")

        domain_judgments: list[str] = []
        for j, col in enumerate(cols):
            if col == "Overall":
                judgment = overall_rating(domain_judgments)
                flagged = False
            else:
                judgment, _agreement, flagged = grid.get(
                    (paper, col), ("No information", "0/0", False)
                )
                domain_judgments.append(judgment)

            color = COLOR.get(judgment, "#cccccc")
            sym = SYMBOL.get(judgment, "?")
            edge = "black" if not flagged else "#7a0010"
            lw = 1.0 if not flagged else 2.5

            circ = patches.Circle((col_xs[j], y), 0.36, facecolor=color,
                                  edgecolor=edge, linewidth=lw)
            ax.add_patch(circ)
            ax.text(col_xs[j], y, sym, fontsize=14, ha="center", va="center",
                    color="white", fontweight="bold")

    # Vertical separator between the domain columns and the Overall column.
    sep_x = (n_dom - 1 + overall_x) / 2
    ax.plot([sep_x, sep_x], [-(n_studies + 0.5), 0.9],
            color="black", linewidth=0.8)

    # --- Bottom area: domain key + legend, stacked vertically (full-width) ---
    bottom_x = -label_w + 0.4
    cursor = -(n_studies + 1.0)

    # Domains key
    ax.text(bottom_x, cursor, "Domains:", fontsize=10, fontweight="bold", va="top")
    cursor -= line_h
    for did in domain_ids:
        ax.text(bottom_x, cursor, f"{did}: {_short_name(domain_names[did])}",
                fontsize=9, va="top")
        cursor -= line_h

    # Spacing between sections
    cursor -= line_h

    # Judgement legend (vertical list, full row width)
    ax.text(bottom_x, cursor, "Judgement", fontsize=10, fontweight="bold", va="top")
    cursor -= line_h
    for label, sym in legend_items:
        cy = cursor - 0.18
        ax.add_patch(patches.Circle((bottom_x + 0.18, cy), 0.18,
                                     facecolor=COLOR.get(label, "#cccccc"),
                                     edgecolor="black", linewidth=0.8))
        ax.text(bottom_x + 0.18, cy, sym, fontsize=9, ha="center", va="center",
                color="white", fontweight="bold")
        ax.text(bottom_x + 0.55, cy, label, fontsize=9, va="center")
        cursor -= line_h

    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(exist_ok=True)
        plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
        print(f"Saved {out_path} ({n_studies} studies × {len(domain_ids)} domains)")
    if show:
        plt.show()
    if return_fig:
        return fig
    plt.close(fig)
    return None


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
