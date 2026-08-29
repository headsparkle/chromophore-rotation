# Figure 5 build pipeline

Generates **Figure 5** of *"Ground-state chromophore geometry, not cage size,
tracks quantum yield in fluorescent proteins"*: a side-by-side comparison of a
**bright, loose-caged** green FP (StayGold, PDB **7Y40**) and a **dim,
tight-caged** green FP (Dronpa-2, PDB **6NQK**), each showing the chromophore,
its first-shell cage residues, position 203, and the sterically accessible
(tau, phi) region from the rotational scan.

Output: `figure5.png` and `figure5.pdf` (300 dpi).

---

## Read this first: does it actually use the GPU?

**No, and it does not need to.** PyMOL's photorealistic renderer (`ray`) is a
**CPU** ray tracer (multithreaded across cores); it does not use CUDA/OpenGL
GPUs. The scan overlay is pure NumPy, also CPU. Running on a GPU node is fine
and gives you a headless environment plus lots of cores (ray tracing scales
with `--cpus-per-task`), but the GPU itself sits idle. If your only spare
compute is a GPU partition, use it; just do not expect a GPU speed-up.

If you specifically want GPU **rasterization** (lower quality than ray tracing,
rarely worth it), replace `cmd.ray(...); cmd.png(..., ray=1)` in
`render_panels.pml` with `cmd.draw(W, H); cmd.png(..., ray=0)` and run under a
GPU EGL context or `xvfb-run` with GPU GLX. The default (ray tracing) is
recommended.

---

## Files

| file | what it does |
|------|--------------|
| `environment.yml`    | conda env (pymol-open-source, gemmi, numpy, scipy, matplotlib) |
| `fetch_structures.sh`| downloads `7y40.cif`, `6nqk.cif` into `structures/` |
| `render_panels.pml`  | PyMOL: renders `panel_bright.png`, `panel_dim.png` (reads local CIFs; used by `run_fig5.sh`) |
| `figure5_chromophore_cage.pml` | standalone self-fetching alternative to `render_panels.pml` for a quick one-off render |
| `make_envelope.py`   | reruns the (tau, phi) scan -> `envelope_7y40.png`, `envelope_6nqk.png` |
| `compose_fig5.py`    | assembles panels + envelopes -> `figure5.png/.pdf` |
| `run_fig5.sh`        | runs all four steps in order |
| `run_fig5.slurm`     | SLURM wrapper for a GPU/compute node |
| `scan/barrel.py`, `scan/rotate.py` | the project's scan core (bundled so the folder is self-contained) |

---

## Quick start

```bash
# 1. copy this whole fig5_pipeline/ folder to the cluster
scp -r fig5_pipeline you@cluster:~/fig5_pipeline

# 2. create the environment once
cd ~/fig5_pipeline
mamba env create -f environment.yml     # or: conda env create -f environment.yml
conda activate fig5

# 3a. interactive node:
bash run_fig5.sh

# 3b. or submit to the GPU/compute queue:
sbatch run_fig5.slurm
```

The result is `figure5.png` / `figure5.pdf` in the same folder.

---

## Step-by-step (if you prefer to run pieces yourself)

```bash
conda activate fig5

# structures
bash fetch_structures.sh                       # -> structures/7y40.cif, 6nqk.cif

# structural panels (head-less PyMOL)
pymol -cq render_panels.pml                     # -> panel_bright.png, panel_dim.png
#   if PyMOL complains about a display/GL context, wrap it:
#   xvfb-run -a pymol -cq render_panels.pml

# accessible-space overlays (NumPy scan, ~seconds each)
python make_envelope.py structures/7y40.cif structures/6nqk.cif

# final composite
python compose_fig5.py                          # -> figure5.png, figure5.pdf
```

---

## Things you may need to adjust

- **Gatekeeper residue number.** `render_panels.pml` colors `resi 203` orange.
  StayGold and Dronpa-family entries are Aequorea/Anthozoan lineages; confirm
  that the residue equivalent to avGFP position 203 is deposited as 203 in each
  file, and set `GK_RESI_BR` / `GK_RESI_DIM` to the aligned number if not.
- **Chromophore residue code.** `render_panels.pml` and the scan detect the
  matured chromophore automatically (by residue-name list, then by "organic,
  non-protein" fallback). If a panel comes out with no teal chromophore, add
  that structure's chromophore 3-letter code to `CHROMO_RESN`.
- **f_allowed labels.** `compose_fig5.py` prints the accessible fractions in the
  per-panel header; `make_envelope.py` also prints the freshly computed value
  for each structure. Update the `meta=` strings in `compose_fig5.py` if you
  want them to match to more decimals.
- **View / orientation.** Tweak `orient`, `zoom`, `turn` in `render_panels.pml`
  to taste, then re-run just that step.

---

## No cluster? It runs on a laptop too

Everything here is CPU-only, so a local machine with the `fig5` conda env works
identically:

```bash
conda activate fig5
bash run_fig5.sh
```

macOS note: PyMOL renders without `xvfb-run` on a Mac; the `run_fig5.sh` guard
skips `xvfb-run` automatically when it is not installed.
