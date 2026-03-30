# iTAK3 Refactor Plan

## Goal

Stabilize the current iTAK3 architecture without changing scientific behavior:

- keep direct mode, `--predict`, and `--list-predict` runnable
- keep TF/TR and PK outputs compatible with the current repository
- reduce mode drift and file-name guessing
- make every large refactor step gated by repeatable regression tests

## Existing test data

### `test_protein.fasta`
- Coverage:
  - mixed default workflow
  - direct mode
  - TF/TR outputs
  - PK-empty path
- Expected:
  - pipeline succeeds
  - TF/TR outputs exist
  - PK outputs exist but may contain zero classified PKs

### `test_protein_kinase.fasta`
- Coverage:
  - positive PK workflow
  - direct mode PK classification
  - prediction mode PK classification
  - debug-mode PK JSON output
- Expected:
  - at least 2 PK classifications
  - `protein_kinase/<name>_pk_classified.fasta` exists
  - `protein_kinase/pk_classification.tsv` exists
  - `protein_kinase/match.json` exists in `--debug`

## Test entrypoint

Use:

```bash
./checkpoint_test.sh --suite quick
./checkpoint_test.sh --suite full
```

### `quick`
- syntax check
- direct mode baseline on `test_protein.fasta`
- positive PK direct-mode regression on `test_protein_kinase.fasta`

### `full`
- everything in `quick`
- positive PK prediction-mode regression
- positive PK debug-mode regression

## Refactor checkpoints

### Checkpoint 0: Freeze baseline

Scope:
- do not change behavior
- only confirm the current baseline and keep tests green

Completion criteria:
- `checkpoint_test.sh --suite full` passes
- no dirty worktree after test-only changes

Required test:

```bash
./checkpoint_test.sh --suite full --label cp0_baseline
```

### Checkpoint 1: Introduce explicit pipeline context

Scope:
- replace ad hoc path guessing with a small internal context object
- centralize:
  - input FASTA
  - processed FASTA
  - project output
  - InterProScan JSON path
  - hmmscan output path
  - TF preclassification paths
  - PK output paths

Completion criteria:
- `run_analysis_modules()` no longer reconstructs state from guessed filenames
- direct mode and predict mode both use the same context builder

Required test:

```bash
./checkpoint_test.sh --suite full --label cp1_context
```

### Checkpoint 2: Extract workflow steps into stable functions

Scope:
- split the large main script into explicit steps:
  - preprocess input
  - run TF prediction
  - extract predicted TF FASTA
  - run PK analysis
  - run InterProScan
  - run hmmscan
  - run TF/TR classification
  - write combined summaries

Completion criteria:
- `predict_transcription_factors()` and `analyze_sequences_directly()` become thin orchestrators
- no duplicated path setup logic between the two modes

Required test:

```bash
./checkpoint_test.sh --suite full --label cp2_steps
```

### Checkpoint 3: Unify success semantics across modes

Scope:
- decouple PK success from TF extraction success
- define mode-specific rules clearly:
  - no TF but valid PK should not be treated as total failure
  - no TF should produce empty TF/TR outputs in a predictable way
  - PK outputs should still be valid

Completion criteria:
- `--predict` handles "PK positive, TF empty" cleanly
- empty-result outputs are deterministic and documented

Required test:

```bash
./checkpoint_test.sh --suite full --label cp3_semantics
```

### Checkpoint 4: Normalize output contracts

Scope:
- define a single internal result schema for TF/TR and PK
- write all output views from that schema
- minimize parallel output writers that can drift independently

Completion criteria:
- TF/TR and PK table/JSON/FASTA writers are generated from normalized records
- `all_match_tbl.txt` is a debug-only combined summary derived from normalized outputs

Required test:

```bash
./checkpoint_test.sh --suite full --label cp4_outputs
```

### Checkpoint 5: Tighten error handling and diagnostics

Scope:
- reduce broad `except Exception`
- stop swallowing module import/runtime errors without context
- standardize failure messages from external tools

Completion criteria:
- import errors, missing files, and subprocess failures report the exact failing stage
- broad exception handlers are limited to top-level CLI boundaries

Required test:

```bash
./checkpoint_test.sh --suite full --label cp5_errors
```

### Checkpoint 6: Bring `--list-predict` back into contract

Scope:
- decide whether PK is supported in `--list-predict`
- if supported, implement once without duplicated reruns
- if not supported, make the output contract explicit and consistent

Completion criteria:
- `--list-predict` behavior matches CLI/documentation
- no hidden mode-specific omissions

Required test:

```bash
./checkpoint_test.sh --suite full --label cp6_list_predict
```

## Working rules during refactor

1. Change only one checkpoint scope at a time.
2. After each checkpoint-sized edit, run the required checkpoint test before proceeding.
3. Do not combine structural refactor and new biological logic in the same checkpoint.
4. Keep legacy output files until the new normalized contract is proven stable.
5. Only remove compatibility outputs after at least one later checkpoint passes in `full` mode.

## Minimum acceptance bar

Before any later push:

```bash
./checkpoint_test.sh --suite full --label pre_push
git status --short
```

Expected:
- all tests pass
- worktree is clean except intended code changes
