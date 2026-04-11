# Changelog

## Unreleased

### Added
- Reintroduced protein kinase identification and classification into iTAK3 using bundled HMM profiles under `db/itak3_pk/`.
- Added a dedicated protein kinase pipeline in `module/protein_kinase.py` with:
  - kinase identification via `PF00069` / `PF07714`
  - Shiu classification
  - PPC classification
  - subclass refinement for WNK1 and MAK groups
- Added bundled protein kinase database assets under `db/itak3_pk/`.
- Added positive protein kinase example input: `test_protein_kinase.fasta`.
- Added positive protein kinase smoke-test coverage for both direct mode and `--predict` mode.
- Added optional combined result summary output: `result/all_match_tbl.txt` in `--debug` mode.

### Changed
- Promoted `itak` to the primary user-facing entrypoint and renamed the internal CLI orchestrator to `itak_cli.py`.
- Removed the deprecated compatibility entrypoints `itak3-v1.0.py` and `run_itak3_local.sh`; repository scripts, install output, and docs now point only to `./itak`.
- Strengthened bundled InterProScan validation in both dependency checks and `install_runtime`, including top-level file/dir checks, dataset checks, and `interproscan.sh -version` self-test coverage.
- Updated `install_runtime` so invalid `db/interproscan` or invalid `--interproscan-dir` inputs no longer hard-fail immediately; the installer can now diagnose the issues, prompt for bundled db repair, or auto-repair via `--download-db`.
- Updated `install_runtime --status` to report invalid or missing InterProScan layouts without crashing, and to clearly mark `./itak` as the preferred CLI.
- Extended `pre_model/predict.py` with explicit `--device` selection and optional batch-progress reporting via `--progress-every`, while preserving the default automatic device choice.
- Integrated protein kinase analysis into both direct analysis mode and prediction mode.
- Added `--skip-pk` to disable protein kinase analysis when needed.
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
- `./smoke_test.sh --input test_protein_kinase.fasta --require-pk 2 --output output/smoke_test_regression_pk`
- `./smoke_test.sh --predict --input test_protein_kinase.fasta --require-pk 2 --output output/smoke_test_predict_pk_recheck`
- `./itak -i test_protein_kinase.fasta --appl PROSITEPROFILES --debug -o output/pk_debug_json`
- `./.venv/bin/python pre_model/predict.py --fasta test_pk_no_tf_candidate.fasta --output output/predict_cli_options_check.csv --device cpu --progress-every 1`
- `python3 tools/install_runtime.py --status`
- `python3 tools/install_runtime.py --status --interproscan-dir /tmp/itak_missing_interproscan`

### Relevant commits
- `2e10ea3` Add protein kinase classification workflow
- `891c080` Add positive protein kinase example input
- `9e90048` Add positive protein kinase smoke test
- `c6262c9` Add combined TF and kinase result table
- `ec9e722` Document predictive kinase smoke test
- `c903a80` Align kinase outputs with TF format
- `15363a8` Add kinase debug JSON output
- `90c4683` Add classified kinase FASTA naming
