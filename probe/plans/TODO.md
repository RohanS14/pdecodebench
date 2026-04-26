# Probe Experiment TODO

## Done

- [x] **Hidden state extraction** — `extract_hidden.py` completed (job 7013364)
  - Output: `probe/hidden_states/Qwen_Qwen2.5-Coder-7B-Instruct.npz`
  - Shape: (96, 29, 3584) for mean_pool and last_tok

- [x] **RSA analysis** — `rsa_analysis.py` completed (job 7013455)
  - Static PNGs saved to `probe/figures/`: layer heatmaps (layers 0,7,14,21,28),
    block score vs. layer, mod-type comparison subplots
  - Both mean_pool and last_tok

- [x] **Pooled probe — mean_pool** — completed (job 7015083, timed out after mean_pool finished)
  - Output: `probe/results/probe_pooled_mean_pool.csv`
  - Static PNGs: per-label accuracy vs. layer + mod-type breakdown

---

## Running Now

- [ ] **Pooled probe — last_tok** — job 7020376, running on cs604
  - Will produce: `probe/results/probe_pooled_last_tok.csv` + figures

- [ ] **Clean-transfer probe — mean_pool + last_tok** — job 7020377, pending
  (blocked on QOSMaxMemoryPerUser — waiting for 7020376 to free memory)
  - Will produce: `probe/results/probe_transfer_mean_pool.csv`
                  `probe/results/probe_transfer_last_tok.csv`

---

## Blocked / Waiting

- [ ] **Interactive HTML report** (`viz_interactive.py` + `probe/slurm/viz.slurm`)
  - Blocked on: all 4 probe CSVs being present
  - Required inputs:
    - `probe/results/probe_pooled_mean_pool.csv` ✓
    - `probe/results/probe_pooled_last_tok.csv` (in progress)
    - `probe/results/probe_transfer_mean_pool.csv` (pending)
    - `probe/results/probe_transfer_last_tok.csv` (pending)
  - Also needs `statsmodels` installed in venv (for Wilson binomial CI)

---

## TODO — After Jobs Complete

- [ ] Verify all 4 CSVs saved cleanly, check for NaN-heavy columns
- [ ] Install `statsmodels` into `probe/venv`:
  ```bash
  source probe/venv/bin/activate && pip install statsmodels
  ```
- [ ] Implement `probe/viz_interactive.py`:
  - Section 1: Overview table (model, N, layers, dim, pooling, dataset distribution)
  - Section 2: RSA — interactive layer heatmap (dropdown), block score vs. layer,
    mod-type subplots
  - Section 3: Pooled probe — accuracy vs. layer with CI bands, mod-type bar chart
    with Wilson CI error bars
  - Section 4: Transfer probe — label × mod_type heatmap, per-label transfer curves
    with CI shading
  - Single self-contained HTML (Plotly JS embedded, no CDN dependency)
- [ ] Write `probe/slurm/viz.slurm` (CPU job, ~10 min) and submit
- [ ] Open `probe/results/report.html` in browser and verify all dropdowns work

---

## Notes

- Probe time limit was bumped to 3h (was 1h, caused job 7015083 to cancel mid-run)
- mean_pool probe takes ~1h; expect last_tok similar; transfer likely similar
- Transfer job stuck in QOSMaxMemoryPerUser — will unblock once pooled job finishes
- All jobs send BEGIN/END/FAIL email to rs9768@nyu.edu
