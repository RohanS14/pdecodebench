#!/bin/bash
# Verify PDE sims + render bounded-invalid cases in the pdecodebench venv on torch.
cd /home/ehb7466/pde-verify || exit 1
rm -f DONE.marker
E=/scratch/ehb7466/envs/pdecodebench
PY=$E/bin/python
export PIP_CACHE_DIR=/scratch/ehb7466/.pip-cache
export MPLBACKEND=Agg

echo "=== [1/5] install matplotlib + jax (scipy already present) ==="
$PY -m pip install --quiet matplotlib 'jax[cpu]' 2>&1 | tail -3

echo "=== [2/5] mpi4py (best effort; needs an MPI module) ==="
source /etc/profile.d/modules.sh 2>/dev/null || true
( module load openmpi 2>/dev/null || module load openmpi/4.1.5 2>/dev/null || module load intel-mpi 2>/dev/null || true
  $PY -m pip install --quiet mpi4py 2>&1 | tail -2 ) || echo "  mpi4py install failed"
$PY -c 'import importlib.util as u; print("  present:", {m: bool(u.find_spec(m)) for m in ["scipy","matplotlib","jax","mpi4py"]})'

echo "=== [3/5] extract codes from v4 ==="
$PY extract_code.py 2>&1 | tail -2

echo "=== [4/5] execution verification (full 128) ==="
$PY verify_simulations.py > verify_report.txt 2>&1 || true
sed -n '/VERIFICATION SUMMARY/,/Anomalies incorrectly/p' verify_report.txt

echo "=== [5/5] render bounded-invalid + NS cases ==="
$PY render_sims.py --xlsx data/pdedata_clean_v4.xlsx --out figs \
  --titles Heat_NoComm_InValid_3 Heat_NoComm_Valid_3 \
           Wave_NoComm_InValid_3 Wave_NoComm_Valid_3 \
           Burgers_Invalid2 Burgers_Valid2 Burgers_Invalid4 Burgers_Valid4 \
           NavierStokes_Invalid3 NavierStokes_Invalid4 2>&1 | tail -20

echo "ALLDONE $(date -u)" > DONE.marker
