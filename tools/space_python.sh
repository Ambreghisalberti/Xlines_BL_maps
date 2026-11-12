#!/bin/bash
# The "space" env lives in the JupyterHub container; reach it through one of my running kernels.
for pid in $(pgrep -u "$USER" -f "envs/space/bin/python -m ipykernel_launcher"); do
  P=/proc/$pid/root/opt/conda/envs/space/bin/python
  if [ -x "$P" ]; then export MPLCONFIGDIR="$(dirname "$0")/.mplconfig"; exec nice -n 19 "$P" "$@"; fi
done
echo "No running space kernel found: open a notebook with the space kernel in JupyterHub." >&2; exit 1
