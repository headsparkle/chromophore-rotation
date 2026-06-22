"""
regen_table2_canonical.py
=========================

Regenerate Table 2 (per-unique-FP median accessible torsional fraction and
barrel minor axis by color class) under ONE canonical unique-FP grouping, so
that it is consistent with the n=38-red QY analysis used elsewhere.

Canonical grouping (decided 2026-06-21):
  - fp_id = fpbase_slug, falling back to seq_match_slug (from lit_qy_curated).
  - Structures without any slug are dropped (Table 2 is the slug-grouped set,
    as the original table and the QY analysis both are).
  - Each fp_id (unique FP) is assigned ONE color = the majority color_class of
    its structures (deterministic tie-break: alphabetical).
  - Per FP: median f_allowed_folded and median barrel_minor_axis.
  - Per color: count of unique FPs, median of the per-FP medians.

Prints the regenerated table next to the current manuscript values and lists
the multi-color slugs that caused the original inconsistency. Does NOT touch
the docx.
"""
import pandas as pd

GEO = "data/merged_for_aggregate.csv"
QY = "data/lit_qy_curated.csv"
MA = "minor_axis"
ORDER = ["blue", "cyan", "green", "yellow", "orange", "red"]
CURRENT = {  # current manuscript Table 2: (n, median frac, median minor)
    "blue": (3, 0.030, 30.4), "cyan": (18, 0.005, 29.4),
    "green": (66, 0.013, 29.9), "yellow": (12, 0.012, 29.5),
    "orange": (6, 0.012, 29.0), "red": (38, 0.006, 28.1),
}


def main():
    df = pd.read_csv(GEO)
    qy = pd.read_csv(QY)
    qy["fp_id"] = qy["fpbase_slug"].fillna(qy["seq_match_slug"])
    m = df.merge(qy[["pdb_id", "fp_id"]], on="pdb_id", how="left")
    m = m.dropna(subset=["fp_id", "f_allowed_folded"])

    # diagnose multi-color slugs
    ncol = m.groupby("fp_id")["color_class"].nunique()
    multi = ncol[ncol > 1]

    def majority_color(s):
        return sorted(s.value_counts().index, key=lambda c: (-s.value_counts()[c], c))[0]

    per = m.groupby("fp_id").agg(
        color=("color_class", majority_color),
        f=("f_allowed_folded", "median"),
        minor=(MA, "median"),
    ).reset_index()

    print(f"Canonical unique-FP grouping: {len(per)} FPs "
          f"({len(multi)} slugs span >1 color, resolved by majority vote)\n")
    print(f"{'color':8} | {'CANONICAL n / frac / minor':30} | {'CURRENT (manuscript)':22} | changed?")
    print("-" * 92)
    rows = {}
    for c in ORDER:
        sub = per[per.color == c]
        n, fr, mi = len(sub), sub.f.median(), sub.minor.median()
        rows[c] = (n, round(fr, 3), round(mi, 1))
        cn, cf, cm = CURRENT[c]
        chg = "  ".join(
            x for x, ok in [
                (f"n {cn}->{n}", n == cn),
                (f"frac {cf}->{fr:.3f}", round(fr, 3) == cf),
                (f"minor {cm}->{mi:.1f}", round(mi, 1) == cm),
            ] if not ok
        ) or "(no change)"
        print(f"{c:8} | n={n:3d}  {fr:.4f}  {mi:5.1f}            | "
              f"n={cn:3d}  {cf:.3f}  {cm:5.1f}        | {chg}")

    print("\nProposed Table 2 cells (rounded as in manuscript):")
    for c in ORDER:
        n, fr, mi = rows[c]
        print(f"  {c.capitalize():7}  ; {n}  | {fr:.3f} | {mi:.1f}")

    print(f"\nMulti-color slugs (n={len(multi)}) -- the source of the 35-vs-38 split:")
    for fp in multi.index:
        cc = m[m.fp_id == fp].color_class.value_counts().to_dict()
        print(f"  {fp:32} {cc}")


if __name__ == "__main__":
    main()
