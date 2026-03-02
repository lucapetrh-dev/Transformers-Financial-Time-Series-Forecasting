from __future__ import annotations

import argparse
from pathlib import Path

from forecast.analysis.paper_figures import generate_paper_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate thesis-ready figure pack from results artifacts")
    parser.add_argument("--results-root", type=str, default="results/paper")
    parser.add_argument("--output-dir", type=str, default="results/paper/overleaf_thesis/Images/generated")
    args = parser.parse_args()

    records = generate_paper_figures(
        results_root=Path(args.results_root),
        output_dir=Path(args.output_dir),
    )
    print(f"Generated {len(records)} figures in {args.output_dir}")
    print(f"Manifest: {Path(args.output_dir) / 'FIGURES_MANIFEST.csv'}")
    for rec in records:
        print(f" - {rec.figure_id}: {rec.filename}")


if __name__ == "__main__":
    main()
