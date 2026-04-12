---
# iTAK3 - Transcription Factor and Protein Kinase Analysis
---

## Overview

iTAK3 is a TF/TR/PK analysis pipeline for protein FASTA inputs. The current design uses a
prediction-first workflow by default:

1. preprocess input sequences into valid proteins
2. predict likely TF/TR candidates with the bundled deep-learning model
3. run InterProScan and hmmscan only on the predicted TF/TR candidates
4. classify TF/TR families with rule-based logic
5. classify protein kinases from the full processed protein FASTA

If needed, users can disable the predictive prefilter with `--no-predict` and force direct
analysis on all eligible sequences.

## Key Features

1. Default predictive prefilter to reduce downstream InterProScan and hmmscan cost
2. Direct fallback mode with `--no-predict`
3. Rule-based TF/TR family classification
4. Protein kinase identification and classification with bundled HMM assets
5. Bundled iTAK-managed InterProScan runtime and slimmed database payload
6. Grad-CAM heatmap generation from the prediction model

## System Requirements

- Python 3.10+
- Java 11+
- Perl
- HMMER 3

## Python Dependencies

The default workflow requires:

- biopython
- pandas
- numpy
- torch
- matplotlib

Install them with:

```bash
pip install -r requirements.txt
```

## Project Layout

```text
itak3/
├── itak
├── itak_cli.py
├── docs/
├── module/
├── pre_model/
│   ├── model.pth
│   ├── predict.py
│   └── supplementary_model/
├── db/
│   ├── interproscan/
│   ├── hmm_pk/
│   └── hmm_self_build/
├── runtime/
├── tools/
├── rule.txt
├── requirements.txt
├── smoke_test.sh
├── checkpoint_test.sh
└── test_protein.fasta
```

## Installation

### Recommended setup: pixi

```bash
pixi run configure-runtime
```

After setup, use:

```bash
./itak
```

or:

```bash
pixi run itak --help
```

Useful pixi tasks:

```bash
pixi run runtime-status
pixi run runtime-check
pixi run check-deps
pixi run smoke-test
pixi run checkpoint-quick
```

### Existing Python / conda / bioconda environment

If you already manage your own environment, install the Python packages there and then configure
the bundled InterProScan runtime:

```bash
pip install -r requirements.txt
python3 tools/configure_interproscan_runtime.py
python3 tools/configure_interproscan_runtime.py --status
python3 tools/configure_interproscan_runtime.py --check
```

## Runtime Policy

- The only official entrypoint is `itak`.
- `itak3-v1.0.py`, `run_itak3_local.sh`, and `tools/install_runtime.py` are retired.
- `install_runtime.sh` is only a migration stub.
- iTAK does not auto-switch into a repository `.venv`.
- iTAK always uses the bundled `db/interproscan`.
- External InterProScan paths are intentionally unsupported.

## Database Policy

### InterProScan

- iTAK does not use the official full InterProScan data package.
- `db/interproscan/` is an iTAK-managed runtime plus slimmed database payload for TF/TR/PK
  workflows.
- This bundled payload is the only supported InterProScan database source.

### TF/TR hmmscan database

- `db/hmm_self_build/self_build.hmm`

### Protein kinase database

- `db/hmm_pk/`

## Documentation

- Current design: [docs/CURRENT_DESIGN.md](/Users/kentnf/projects/cornell/itak2_9.19/docs/CURRENT_DESIGN.md)
- Documentation index: [docs/README.md](/Users/kentnf/projects/cornell/itak2_9.19/docs/README.md)

## Usage

### Syntax

```bash
itak [options] -i <input.fasta>
```

Repository-local usage:

```bash
./itak [options] -i <input.fasta>
```

### Primary arguments

- `-i, --input`: input FASTA file
- `-o, --output`: output directory
- `-t, --threshold`: prediction threshold in `[0, 1]` for TF/TR prefiltering
- `--appl`: InterProScan application list
- `--score`: score threshold for InterProScan and hmmscan-derived domain filtering
- `--classification-mode`: `specific` or `score`
- `--skip-pk`: skip protein kinase classification
- `--debug`: write additional intermediate outputs

### Workflow selection

- default: prediction-first workflow
- `--no-predict`: direct fallback mode
- `--predict`: legacy alias for the default workflow
- `--list-predict`: run prediction once and emit multiple threshold-specific outputs

### Model-related options

- `--predict-mode`: `fast` or `full`
- `--use-supplementary`: add supplementary prediction models
- `--supplementary-only`: skip the main model and use supplementary models only
- `--supp-models`: explicit supplementary model paths
- `--grad-cam-mode`: `none`, `all`, `positive`, or `fast`

### Validation and test options

- `--check-deps`: validate the default runtime and exit
- `--skip-deps-check`: skip dependency validation
- `-test, --test-mode`: use existing InterProScan JSON and hmmscan outputs
- `-json, --json-file`: InterProScan JSON for test mode
- `-spechmm, --spechmm-file`: hmmscan result file for test mode

## Examples

### Default predictive workflow

```bash
itak -i test_protein.fasta
```

`test_protein.fasta` is the only bundled test FASTA. It is a mixed sample containing TF/TR, PK,
and non-regulatory proteins together.

### Default workflow with an explicit threshold

```bash
itak -i test_protein.fasta -t 0.1
```

### Direct fallback workflow

```bash
itak --no-predict -i test_protein.fasta
```

### Specify an output directory

```bash
itak -i test_protein.fasta -o /path/to/output
```

### Restrict InterProScan applications

```bash
itak -i test_protein.fasta --appl CDD,Pfam,SMART
```

### Use supplementary models

```bash
itak --use-supplementary -i input.fasta
itak --supplementary-only -i input.fasta
itak --use-supplementary --supp-models a.pth b.pth -i input.fasta
```

### Grad-CAM

```bash
itak --grad-cam-mode fast -i input.fasta
itak --grad-cam-mode all -i input.fasta
```

### Test mode

```bash
itak -test -i input.fasta -json ipr_result.json -spechmm hmmscan_result.tbl
```

### Dependency check

```bash
itak --check-deps
```

### Smoke tests

```bash
./smoke_test.sh
./smoke_test.sh --no-predict
./smoke_test.sh --require-pk 2
./checkpoint_test.sh --suite quick
./checkpoint_test.sh --suite full
```

## Outputs

The pipeline writes results under the chosen output directory:

```text
<project_output>/
├── protein_model_preclassification/
│   ├── <name>_prediction.csv
│   ├── <name>_prediction_tf_only.csv
│   └── <name>_tf_sequences.fasta
├── InterproScan/
│   └── <input>.json
├── hmmscan/
│   └── result.tbl
├── protein_kinase/
│   ├── <name>_pk_classified.fasta
│   ├── pk_classification.tsv
│   ├── shiu_classification.txt
│   ├── PPC_classification.txt
│   └── match.json
├── getrule.json
└── result/
    ├── match_tbl.txt
    ├── all_match_tbl.txt
    ├── <name>_tf_classified.fasta
    ├── match.json
    ├── processed_ipr_domains.json
    └── pfamspec.json
```

Notes:

- `protein_model_preclassification/` is produced by the default predictive workflow, `--list-predict`,
  or Grad-CAM runs.
- `match.json`, `all_match_tbl.txt`, `getrule.json`, `processed_ipr_domains.json`, and `pfamspec.json`
  are debug-oriented outputs and may be absent unless `--debug` is enabled.

## Workflow Summary

1. Validate the input FASTA
2. Validate runtime dependencies
3. Preprocess input sequences into proteins
4. Run the default predictive prefilter unless `--no-predict` is used
5. Run protein kinase classification on the processed protein FASTA
6. Run InterProScan and hmmscan for TF/TR analysis
7. Apply rule-based TF/TR classification
8. Write reports, tables, and classified FASTA outputs
