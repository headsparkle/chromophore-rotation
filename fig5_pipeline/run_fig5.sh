#!/usr/bin/env bash
# End-to-end Figure 5 build. Run after `conda activate fig5`.
# Works on any Linux node (login, compute, or GPU); the GPU is not
# required (PyMOL ray tracing is CPU-multithreaded).
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/4  fetch structures =="
bash fetch_structures.sh

echo "== 2/4  render structural panels (PyMOL, headless) =="
# PyMOL open-source usually ray-traces fine under -cq. If your build errors
# with a GL/context message, wrap it in xvfb-run (provides a virtual display):
if command -v xvfb-run >/dev/null 2>&1; then
  xvfb-run -a --server-args="-screen 0 1920x1080x24" pymol -cq render_panels.pml
else
  pymol -cq render_panels.pml
fi

echo "== 3/4  compute (tau, phi) accessible-space envelopes (NumPy) =="
python make_envelope.py structures/7y40.cif structures/6nqk.cif

echo "== 4/4  compose final figure =="
python compose_fig5.py

echo "Done. Outputs: figure5.png, figure5.pdf (plus panel_*.png, envelope_*.png)."
