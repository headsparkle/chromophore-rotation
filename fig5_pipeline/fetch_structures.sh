#!/usr/bin/env bash
# Download the two structures for Figure 5 into ./structures/ as mmCIF.
# Run on a node with internet, or run locally and copy the folder to the cluster.
set -euo pipefail

mkdir -p structures
cd structures

# StayGold (bright, loose cage) and Dronpa-2 (dim, tight cage)
for id in 7y40 6nqk; do
  if [ ! -s "${id}.cif" ]; then
    echo "fetching ${id} ..."
    curl -fL "https://files.rcsb.org/download/${id}.cif" -o "${id}.cif"
  else
    echo "${id}.cif already present"
  fi
done

echo "done:"
ls -la
