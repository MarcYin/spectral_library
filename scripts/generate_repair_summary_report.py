#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def load_totals(final_root: Path) -> pd.DataFrame:
    frame = pd.read_csv(final_root / "tabular" / "spectra_metadata.csv", usecols=["source_id", "spectrum_id"])
    totals = (
        frame.groupby("source_id", dropna=False)["spectrum_id"]
        .nunique()
        .rename("total_spectra")
        .reset_index()
        .sort_values("source_id")
        .reset_index(drop=True)
    )
    return totals


def stage_frame(
    stage_name: str,
    csv_path: Path,
    spectrum_col: str = "spectrum_id",
    source_col: str = "source_id",
    filter_expr: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    if filter_expr:
        frame = frame.query(filter_expr)
    if frame.empty:
        return pd.DataFrame(columns=["stage", "source_id", "spectrum_id"])
    return frame[[source_col, spectrum_col]].drop_duplicates().assign(stage=stage_name)


def merge_with_totals(frame: pd.DataFrame, totals: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(totals, on="source_id", how="left")
    merged["total_spectra"] = merged["total_spectra"].fillna(0).astype(int)
    merged["affected_pct"] = merged["affected_spectra"] / merged["total_spectra"]
    return merged


def workflow_lines() -> list[str]:
    return [
        "1. Fetch raw source assets and tidy them into `metadata/`, `docs/`, and `data/`.",
        "2. Parse source-specific formats and normalize spectra onto the shared `400-2500 nm` / `1 nm` grid.",
        "3. Filter normalized spectra by coverage threshold and retain only spectra with coverage `>= 0.8`.",
        "4. Apply user-directed curation decisions: remove rejected sources and keep landcover labels as annotations rather than hard filters.",
        "5. Apply targeted source repairs: EMIT absorption artifacts, Santa Barbara tail drift, Ghisacasia targeted deep-band/tail repair, and visible-band robust smoother repair.",
        "6. Re-evaluate source-specific absorption smoothing and select the best final variant per spectrum/window from `base`, `old`, and `guarded` candidates.",
        "7. Recompute landcover annotations and export the final SIAC spectral library package.",
    ]


def build_report(
    final_root: Path,
    siac_root: Path,
    emit_root: Path,
    ghisacasia_root: Path,
    visible_root: Path,
    hybrid_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    totals = load_totals(final_root)

    stage_inputs = [
        (
            "emit_santa_repair",
            stage_frame("emit_santa_repair", emit_root / "diagnostics" / "artifact_repairs.csv"),
            "Blended repair of EMIT second absorption artifacts and Santa Barbara >2400 nm tail drift.",
        ),
        (
            "ghisacasia_targeted_repair",
            stage_frame("ghisacasia_targeted_repair", ghisacasia_root / "diagnostics" / "ghisacasia_repairs.csv"),
            "Targeted Ghisacasia deep-band and tail repair before later source-specific smoothing selection.",
        ),
        (
            "visible_blend_repair",
            stage_frame(
                "visible_blend_repair",
                visible_root / "diagnostics" / "spectrum_flag_counts.csv",
                filter_expr="replaced_visible_band_count > 0",
            ),
            "Visible-window repair using robust smoother detection followed by blended replacement only on flagged spectra.",
        ),
        (
            "source_specific_absorption_final",
            stage_frame(
                "source_specific_absorption_final",
                hybrid_root / "diagnostics" / "variant_selection.csv",
                filter_expr="selected_variant == 'old'",
            ),
            "Final source-specific absorption smoothing actually retained in the hybrid output relative to the pre-smoothing base.",
        ),
    ]

    per_stage_rows: list[dict[str, object]] = []
    stage_summary_rows: list[dict[str, object]] = []
    stage_union_frames: list[pd.DataFrame] = []

    for stage_name, stage_pairs, description in stage_inputs:
        if stage_pairs.empty:
            affected = pd.DataFrame(columns=["source_id", "affected_spectra"])
            stage_spectra = 0
        else:
            stage_union_frames.append(stage_pairs[["source_id", "spectrum_id"]].drop_duplicates())
            affected = (
                stage_pairs.groupby("source_id", dropna=False)["spectrum_id"]
                .nunique()
                .rename("affected_spectra")
                .reset_index()
            )
            stage_spectra = int(stage_pairs["spectrum_id"].nunique())
        merged = merge_with_totals(affected, totals)
        merged.insert(0, "stage", stage_name)
        per_stage_rows.append(merged)
        stage_summary_rows.append(
            {
                "stage": stage_name,
                "description": description,
                "affected_spectra": stage_spectra,
                "affected_pct_of_dataset": stage_spectra / int(totals["total_spectra"].sum()),
                "affected_source_count": int(affected["source_id"].nunique()) if not affected.empty else 0,
            }
        )

    repair_impact_by_source = pd.concat(per_stage_rows, ignore_index=True).sort_values(
        ["stage", "affected_pct", "affected_spectra", "source_id"],
        ascending=[True, False, False, True],
    )

    all_impacts = pd.concat(stage_union_frames, ignore_index=True).drop_duplicates()
    overall = (
        all_impacts.groupby("source_id", dropna=False)["spectrum_id"]
        .nunique()
        .rename("affected_spectra")
        .reset_index()
    )
    repair_impact_overall = merge_with_totals(overall, totals).sort_values(
        ["affected_pct", "affected_spectra", "source_id"],
        ascending=[False, False, True],
    )

    hybrid_selection = pd.read_csv(hybrid_root / "diagnostics" / "variant_selection.csv")
    hybrid_variant_summary = (
        hybrid_selection.groupby(["selected_variant", "source_id"], dropna=False)
        .size()
        .rename("selection_count")
        .reset_index()
        .sort_values(["selected_variant", "selection_count", "source_id"], ascending=[True, False, True])
    )
    hybrid_selection_counts = hybrid_selection["selected_variant"].value_counts().to_dict()

    stage_summary = pd.DataFrame(stage_summary_rows)
    final_summary = {
        "final_root": str(final_root),
        "siac_root": str(siac_root),
        "total_spectra": int(totals["total_spectra"].sum()),
        "source_count": int(totals["source_id"].nunique()),
        "any_repair_affected_spectra": int(len(all_impacts)),
        "any_repair_affected_pct": float(len(all_impacts) / int(totals["total_spectra"].sum())),
        "hybrid_selection_counts": {key: int(value) for key, value in hybrid_selection_counts.items()},
        "top_overall_sources": repair_impact_overall.head(15).to_dict(orient="records"),
    }
    return repair_impact_by_source, repair_impact_overall, stage_summary, hybrid_variant_summary, final_summary


def to_markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    subset = frame.loc[:, columns].copy()
    for column in subset.columns:
        if subset[column].dtype.kind == "f":
            subset[column] = subset[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in subset.iterrows()]
    return "\n".join([header, sep, *rows])


def write_report(
    output_dir: Path,
    final_root: Path,
    siac_root: Path,
    repair_impact_by_source: pd.DataFrame,
    repair_impact_overall: pd.DataFrame,
    stage_summary: pd.DataFrame,
    hybrid_variant_summary: pd.DataFrame,
    final_summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    repair_impact_by_source.to_csv(output_dir / "repair_impact_by_source.csv", index=False)
    repair_impact_overall.to_csv(output_dir / "repair_impact_overall_by_source.csv", index=False)
    stage_summary.to_csv(output_dir / "repair_stage_summary.csv", index=False)
    hybrid_variant_summary.to_csv(output_dir / "hybrid_variant_selection_summary.csv", index=False)
    (output_dir / "repair_summary_report.json").write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    top_overall = repair_impact_overall.head(15).copy()
    stage_top = repair_impact_by_source.groupby("stage", group_keys=False).head(8)

    lines: list[str] = []
    lines.append("# Repair Summary Report")
    lines.append("")
    lines.append(f"- Final normalized root: `{final_root}`")
    lines.append(f"- Final SIAC package: `{siac_root}`")
    lines.append(f"- Total spectra: `{final_summary['total_spectra']}`")
    lines.append(f"- Sources: `{final_summary['source_count']}`")
    lines.append(
        f"- Spectra affected by at least one retained repair stage: `{final_summary['any_repair_affected_spectra']}` "
        f"(`{final_summary['any_repair_affected_pct']:.4f}` of the dataset)"
    )
    lines.append("")
    lines.append("## Processing Workflow")
    lines.extend(workflow_lines())
    lines.append("")
    lines.append("## Stage Summary")
    lines.append(
        to_markdown_table(
            stage_summary,
            ["stage", "affected_spectra", "affected_pct_of_dataset", "affected_source_count"],
        )
    )
    lines.append("")
    lines.append("## Hybrid Source-Specific Absorption Selection")
    lines.append(
        f"- Selected older repaired variant: `{final_summary['hybrid_selection_counts'].get('old', 0)}` window decisions"
    )
    lines.append(
        f"- Reverted to original base spectrum: `{final_summary['hybrid_selection_counts'].get('base', 0)}` window decisions"
    )
    lines.append(
        f"- Selected guarded variant: `{final_summary['hybrid_selection_counts'].get('guarded', 0)}` window decisions"
    )
    lines.append("")
    lines.append(
        to_markdown_table(
            hybrid_variant_summary.head(20),
            ["selected_variant", "source_id", "selection_count"],
        )
    )
    lines.append("")
    lines.append("## Overall Source Impact")
    lines.append(
        to_markdown_table(
            top_overall,
            ["source_id", "affected_spectra", "total_spectra", "affected_pct"],
        )
    )
    lines.append("")
    lines.append("## Top Sources By Stage")
    lines.append(
        to_markdown_table(
            stage_top,
            ["stage", "source_id", "affected_spectra", "total_spectra", "affected_pct"],
        )
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `source_specific_absorption_final` reflects only the final retained source-specific absorption changes in the hybrid output.")
    lines.append("- Window decisions that reverted to `base` are not counted as repaired spectra in the final retained impact totals.")
    lines.append("- Earlier intermediate `v18` and `v19` source-specific smoothing summaries are superseded by this report.")
    (output_dir / "repair_summary_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the final repair summary report against the hybrid v20 output.")
    parser.add_argument("--final-root", required=True)
    parser.add_argument("--siac-root", required=True)
    parser.add_argument("--emit-root", required=True)
    parser.add_argument("--ghisacasia-root", required=True)
    parser.add_argument("--visible-root", required=True)
    parser.add_argument("--hybrid-root", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    final_root = Path(args.final_root)
    siac_root = Path(args.siac_root)
    emit_root = Path(args.emit_root)
    ghisacasia_root = Path(args.ghisacasia_root)
    visible_root = Path(args.visible_root)
    hybrid_root = Path(args.hybrid_root)
    output_dir = Path(args.output_dir) if args.output_dir else final_root / "diagnostics"

    repair_impact_by_source, repair_impact_overall, stage_summary, hybrid_variant_summary, final_summary = build_report(
        final_root,
        siac_root,
        emit_root,
        ghisacasia_root,
        visible_root,
        hybrid_root,
    )
    write_report(
        output_dir,
        final_root,
        siac_root,
        repair_impact_by_source,
        repair_impact_overall,
        stage_summary,
        hybrid_variant_summary,
        final_summary,
    )
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
