# Changelog

## Unreleased

### Added
- Reintroduced protein kinase identification and classification into iTAK3 using bundled HMM profiles under `db/hmm_pk/`.
- Added a dedicated protein kinase pipeline in `module/protein_kinase.py` with:
  - kinase identification via `PF00069` / `PF07714`
  - Shiu classification
  - PPC classification
  - subclass refinement for WNK1 and MAK groups
- Added bundled protein kinase database assets under `db/hmm_pk/`.
- Added positive protein kinase smoke-test coverage for both the default predictive mode and the direct fallback mode.
- Added optional combined result summary output: `result/all_match_tbl.txt` in `--debug` mode.

### Changed
- Promoted `itak` to the primary user-facing entrypoint and renamed the internal CLI orchestrator to `itak_cli.py`.
- Removed the deprecated compatibility entrypoints `itak3-v1.0.py` and `run_itak3_local.sh`; repository scripts, install output, and docs now point only to `./itak`.
- Removed the old `.venv` auto-reexec behavior from `itak`; the entrypoint now runs in the current Python environment instead of silently switching interpreters.
- Strengthened bundled InterProScan validation in both dependency checks and `install_runtime`, including top-level file/dir checks, dataset checks, and `interproscan.sh -version` self-test coverage.
- Updated `install_runtime` so invalid `db/interproscan` or invalid `--interproscan-dir` inputs no longer hard-fail immediately; the installer can now diagnose the issues, prompt for bundled db repair, or auto-repair via `--download-db`.
- Updated `install_runtime --status` to report invalid or missing InterProScan layouts without crashing, and to clearly mark `./itak` as the preferred CLI.
- Added an explicit InterProScan runtime configurator at `tools/configure_interproscan_runtime.py` plus a runtime manifest under `runtime/interproscan_manifest.json`.
- Added an initial `pixi.toml` with explicit tasks for runtime configuration, runtime status checks, dependency checks, smoke tests, and `itak`.
- Retired `install_runtime.sh`; it now exits with migration guidance instead of bootstrapping a `.venv`.
- Removed `tools/install_runtime.py` from the main setup workflow; explicit runtime configuration now goes through `pixi` tasks or `tools/configure_interproscan_runtime.py`.
- Removed support for user-specified external InterProScan paths; iTAK now always uses the bundled `db/interproscan` runtime.
- Extended `pre_model/predict.py` with explicit `--device` selection and optional batch-progress reporting via `--progress-every`, while preserving the default automatic device choice.
- Integrated protein kinase analysis into both direct analysis mode and prediction mode.
- Switched the CLI to a prediction-first default workflow; direct all-sequence analysis is now the explicit `--no-predict` fallback.
- Collapsed Python dependency files into a single `requirements.txt` and aligned `pixi.toml` with the default predictive runtime.
- Merged the main bundled test sample into a mixed TF/TR/PK fixture so default smoke and checkpoint runs better match real proteome-style inputs.
- Removed extra checked-in test FASTA files; targeted regression subsets are now generated from the single mixed sample at runtime.
- Removed the user-facing `--predict` and `--skip-pk` switches; prediction is now always the default entry path and protein kinase classification always runs.
- Hid `--list-predict` from normal CLI help and documentation while retaining it for developer regression coverage.
- Changed the default output directory naming to `<input_basename>_output` in the current working directory.
- Raised the default prediction threshold from `0.1` to `0.5`.
- Added a unified user-facing `--cpu` parameter and wired it through prediction, InterProScan, hmmscan, and protein kinase classification, with automatic capping at the machine CPU-thread limit.
- Updated `smoke_test.sh` and `checkpoint_test.sh` to accept `--cpu` and forward it to `itak`.
- Consolidated project documentation under `docs/`, added a canonical current-design document, and moved transitional planning notes into `docs/archive/`.
- Moved the TF/TR self-build HMM database into `db/hmm_self_build/` and renamed the protein kinase HMM bundle directory to `db/hmm_pk/`.
- Simplified protein kinase outputs to avoid duplicate files:
  - `protein_kinase/<name>_pk_classified.fasta`
  - `protein_kinase/pk_classification.tsv`
  - `protein_kinase/shiu_classification.txt`
  - `protein_kinase/PPC_classification.txt`
  - `protein_kinase/match.json` in `--debug` mode
- Applied `--score` filtering to both InterProScan-derived and hmmscan-derived hits.
- Replaced blind six-frame translation of nucleotide inputs with complete-ORF extraction during preprocessing.
- Enforced processed-FASTA validation so protein sequences containing `*` do not reach InterProScan.

### Verification
- `./checkpoint_test.sh --suite quick`
- `./checkpoint_test.sh --suite full`
- `./checkpoint_test.sh --suite quick --label post_install_runtime`
- `./checkpoint_test.sh --suite full --label final_regression`
- `./checkpoint_test.sh --suite full --label no_legacy_entrypoints`
- `./smoke_test.sh --output output/smoke_test_regression_default`
- `./smoke_test.sh --output output/smoke_test_post_install_runtime`
- `./smoke_test.sh --no-predict --output output/smoke_test_direct_fallback`
- `./smoke_test.sh --require-pk 2 --output output/smoke_test_regression_pk`
- `./smoke_test.sh --require-pk 2 --output output/smoke_test_predict_pk_recheck`
- `./itak -i test_protein.fasta --appl PROSITEPROFILES --debug -o output/pk_debug_json`
- `./.venv/bin/python pre_model/predict.py --fasta output/checkpoints/default_predict_workflow/generated_fixtures/test_pk_no_tf_candidate.fasta --output output/predict_cli_options_check.csv --device cpu --progress-every 1`
- `python3 tools/configure_interproscan_runtime.py --status`
- `python3 tools/configure_interproscan_runtime.py --check`
- `pixi task list`
- `pixi run runtime-status`
- `pixi run configure-runtime`
- `pixi run runtime-check`
- `pixi run -- ./itak --cpu 999 -t 0.3 -i output/checkpoints/cli_surface_cleanup/generated_fixtures/test_pk_no_tf_candidate.fasta --appl PROSITEPROFILES -o output/cpu_cap_predict_check`
- `pixi run -- ./itak --no-predict --cpu 999 -i output/checkpoints/cli_surface_cleanup/generated_fixtures/test_pk_no_tf_candidate.fasta --appl PROSITEPROFILES -o output/cpu_cap_direct_check`
- `bash -n smoke_test.sh checkpoint_test.sh`

### Relevant commits
- `2e10ea3` Add protein kinase classification workflow
- `891c080` Add positive protein kinase example input
- `9e90048` Add positive protein kinase smoke test
- `c6262c9` Add combined TF and kinase result table
- `ec9e722` Document predictive kinase smoke test
- `c903a80` Align kinase outputs with TF format
- `15363a8` Add kinase debug JSON output
- `90c4683` Add classified kinase FASTA naming
