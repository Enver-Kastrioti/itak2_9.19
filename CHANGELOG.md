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
- Added combined result summary output: `result/all_match_tbl.txt`.

### Changed
- Integrated protein kinase analysis into both direct analysis mode and prediction mode.
- Added `--skip-pk` to disable protein kinase analysis when needed.
- Aligned protein kinase outputs with TF/TR result style:
  - `protein_kinase/match_tbl.txt`
  - `protein_kinase/match.json` in `--debug` mode
  - `protein_kinase/<name>_pk_classified.fasta`
- Preserved legacy-compatible protein kinase outputs:
  - `protein_kinase/pk_sequence.fasta`
  - `protein_kinase/pk_classification.tsv`
  - `protein_kinase/shiu_classification.txt`
  - `protein_kinase/PPC_classification.txt`

### Verification
- `./smoke_test.sh --output output/smoke_test_regression_default`
- `./smoke_test.sh --input test_protein_kinase.fasta --require-pk 2 --output output/smoke_test_regression_pk`
- `./smoke_test.sh --predict --input test_protein_kinase.fasta --require-pk 2 --output output/smoke_test_predict_pk_recheck`
- `./run_itak3_local.sh -i test_protein_kinase.fasta --appl PROSITEPROFILES --debug -o output/pk_debug_json`

### Relevant commits
- `2e10ea3` Add protein kinase classification workflow
- `891c080` Add positive protein kinase example input
- `9e90048` Add positive protein kinase smoke test
- `c6262c9` Add combined TF and kinase result table
- `ec9e722` Document predictive kinase smoke test
- `c903a80` Align kinase outputs with TF format
- `15363a8` Add kinase debug JSON output
- `90c4683` Add classified kinase FASTA naming
