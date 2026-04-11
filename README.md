---
# iTAK3 - Transcription Factor and Protein Kinase Analysis
---

Overview
--------
iTAK3 is a bioinformatics tool for transcription factor (TF) prediction and analysis. It integrates
deep-learning-based preclassification with conventional sequence/domain analyses to identify and
classify TFs from protein sequences. The current workflow also includes protein kinase (PK)
identification and classification with bundled HMM profiles.

Key Features
--------
1. TF prediction: use a deep learning model to predict whether protein sequences are potential TF/TR sequences
2. Domain analysis: run InterProScan and hmmscan for domain annotation
3. TF family classification: assign TF families based on rule-based logic
4. Protein kinase analysis: identify kinase domains and classify PKs with bundled Shiu/PPC models
5. Output reporting: generate detailed reports and classified sequence FASTA files

System Requirements
--------
- Python 3.7+
- Java 11+ (required by InterProScan)
- HMMER 3.0+ (hmmscan)

Dependencies
------
Required Python packages:
- Biopython (Bio)
- pandas
- numpy
- Standard library: json, csv, argparse, subprocess, pathlib, os, sys, time, datetime, shutil

Optional Python packages:
- PyTorch (required only for prediction)

External tools:
- hmmscan (HMMER)
- java (required by InterProScan)

Project Layout
--------
## Key Scripts
- `itak`: user-facing CLI entry point
- `itak_cli.py`: internal CLI orchestrator (analysis / prediction / Grad-CAM orchestration)
- `pre_model/predict.py`: deep-learning preclassification (+ Grad-CAM heatmaps)
- `module/`: pipeline modules (FASTA processing, IPR JSON parsing, hmmscan processing, classification, checks)
- `rule.txt`: TF family classification rules

```text
itak3/
├── itak
├── itak_cli.py
├── module/
│   ├── check_dependencies.py
│   ├── validate_fasta.py
│   ├── classification.py
│   ├── get_fasta.py
│   ├── get_json.py
│   ├── get_rule.py
│   └── selfbuild_hmm.py
├── pre_model/
│   ├── model.pth
│   ├── predict.py
│   └── supplementary_model/
├── db/
│   ├── interproscan/
│   └── itak3_pk/
├── hmm/
│   └── self_build.hmm
├── rule.txt
├── output/
├── temp/
├── test_protein.fasta
└── test_protein_kinase.fasta
```

Installation
--------
## Requirements
- Python 3.7+
- Java 11+ (InterProScan)
- HMMER 3.0+ (`hmmscan`)
- PyTorch (only if you use `--predict` / `--list-predict` / `--grad-cam-mode`)

## Recommended setup: pixi
```bash
pixi run configure-runtime
```

Primary entrypoint after setup:
```bash
./itak
```

You can also run the CLI through the managed pixi environment:
```bash
pixi run itak --help
```

The `itak` entrypoint now uses the current Python environment directly. It no longer auto-reexecs into a repository `.venv`.

Common pixi tasks:
```bash
# inspect current runtime status
pixi run runtime-status

# validate engine + iTAK data + minimal self-test without writing config
pixi run runtime-check

# run dependency checks through the managed environment
pixi run check-deps

# run the repository smoke test directly
pixi run smoke-test

# quick regression suite
pixi run checkpoint-quick
```

`pixi run configure-runtime` will:
- generate `db/interproscan/interproscan.local.properties`
- validate the InterProScan engine layout separately from the iTAK-managed data layout
- configure the selected engine to use the iTAK-managed `db/interproscan/data`
- activate iTAK helper binaries on macOS when required
- run a minimal `interproscan.sh -version` self-test
- record the Python and Perl executables from the current pixi environment

You can configure against an external engine while still forcing iTAK to use its own slimmed data:
```bash
pixi run configure-runtime -- --engine-dir /path/to/interproscan
```

Repository-local analysis with an explicitly provided external engine still uses the `itak` entrypoint:
```bash
./itak --interproscan /path/to/interproscan.sh -i input.fasta
```

## Alternative setup: existing Python / conda / bioconda environment
If you already created an environment yourself, install the Python packages there and run the runtime configurator explicitly.

```bash
pip install -r requirements-core.txt
```

Optional prediction packages:
```bash
# CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Apple Silicon / Metal
pip install torch torchvision torchaudio

# GPU (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then configure the runtime:
```bash
python3 tools/configure_interproscan_runtime.py
python3 tools/configure_interproscan_runtime.py --status
python3 tools/configure_interproscan_runtime.py --check
```

External engine example:
```bash
python3 tools/configure_interproscan_runtime.py --engine-dir /path/to/interproscan
```

## Deprecated bootstrap script
- `install_runtime.sh` is retired and now exits with a migration message.
- `tools/install_runtime.py` has been removed from the recommended workflow; use `pixi` or `tools/configure_interproscan_runtime.py` instead.

## InterProScan Database (db/)
- iTAK does not use the official full InterProScan data package.
- `db/interproscan/data` is an iTAK-managed slimmed dataset for TF/TR/PK workflows and must remain under iTAK control.
- If you use an external InterProScan engine via `--interproscan /path/to/interproscan.sh`, keep using the iTAK-managed data directory.

## Protein kinase database (`db/itak3_pk`)
- iTAK3 bundles a reduced protein kinase HMM database under `db/itak3_pk/`.
- This PK workflow uses `hmmscan` only; it does not depend on InterProScan.
- Required profiles include `Tfam_domain.hmm`, `Plant_Pkinase_fam.hmm`, `PlantsPHMM3_89.hmm`,
  `Pkinase_sub_WNK1.hmm`, `Pkinase_sub_MAK.hmm`, plus `GA_table.txt` and `PK_class_desc.txt`.

## Deploy / Install from GitHub
```bash
git clone https://github.com/Enver-Kastrioti/itak2_9.19.git
```

Usage
--------

## Entry point
- Installed or repository-local usage should go through `itak`.
- In a cloned repository, invoke it as `./itak` or `pixi run itak`.

## Syntax
```bash
itak [options] -i <input_file>
```

Repository-local invocation after cloning the repo:
```bash
./itak [options] -i <input_file>
```

## Primary arguments
- `--version`: Show the CLI version and exit
- `-i, --input`: Input FASTA file path (required except with --check-deps)
- `-o, --output`: Output directory path (optional; default: output/<input_basename>/)
- `-t, --threshold`: Prediction threshold in [0,1] (default: 0.1; affects prediction only)
- `--score`: Score threshold for InterProScan and hmmscan-derived domain-hit filtering (default: 1.0)
- `--classification-mode`: Classification mode: specific (specificity-first) or score (score-first, default)

Feature options:
- -test, --test-mode      Test mode: skip InterProScan/hmmscan and validate classification using existing result files
- -json, --json-file      InterProScan JSON result file for test mode
- -spechmm, --spechmm-file hmmscan result file (result.tbl) for test mode
- --predict               Enable prediction: run model inference, then analyze predicted TF sequences
- --predict-mode          Prediction splitting mode: fast (default) or full
- --use-supplementary     Enable supplementary models (optional)
- --supplementary-only    Use supplementary models only; skip the main model
- --supp-models           Optional: specify supplementary model files (multiple allowed); if omitted, use all models in the directory
- --grad-cam-mode         Grad-CAM mode: none (default)/fast/all/positive (all/positive require --predict; --list-predict still unsupported)
- --appl                  InterProScan application list (comma-separated; defaults to common libraries)
- --interproscan          Path to an external interproscan.sh (hmmscan will follow that InterProScan)
- --skip-pk               Skip protein kinase identification/classification
- --debug                 Enable debug outputs (default: off)

Dependency checks:
--check-deps            Check dependencies and exit
--skip-deps-check       Skip dependency checks (not recommended)

Examples
--------

1. Basic analysis (analyze input directly):
   itak -i test_protein.fasta

2. Prediction workflow:
   itak --predict -i test_protein.fasta -t 0.1

3. Specify an output directory:
   itak --predict -i test_protein.fasta -o /path/to/output

4. Enable debug mode:
   itak --predict -i test_protein.fasta --debug

5. Restrict InterProScan applications:
   itak -i test_protein.fasta --appl CDD,Pfam,SMART

6. Test mode:
   itak -test -i input.fasta -json ipr_result.json -spechmm hmmscan_result.tbl

7. Check dependencies:
   itak --check-deps

8. Use supplementary models (requires --predict):
   itak --predict --use-supplementary -i input.fasta
   itak --predict --supplementary-only -i input.fasta
   itak --predict --use-supplementary --supp-models a.pth b.pth -i input.fasta

9. Grad-CAM heatmaps (requires --predict):
   - fast: runs after classification, using the classified TF FASTA under result/ (batch mode)
   - all: runs on the original input FASTA (all sequences)
   itak --predict --grad-cam-mode fast -i input.fasta
   itak --predict --grad-cam-mode all -i input.fasta

10. Skip protein kinase analysis:
   itak -i test_protein.fasta --skip-pk

11. Run a positive protein kinase example:
   itak -i test_protein_kinase.fasta --skip-deps-check

12. Run a positive protein kinase smoke test:
   ./smoke_test.sh --input test_protein_kinase.fasta --require-pk 2

13. Run prediction mode with a positive protein kinase sample:
   ./smoke_test.sh --predict --input test_protein_kinase.fasta --require-pk 2

14. Run list-predict mode with shared protein kinase analysis across thresholds:
   itak --list-predict -i test_pk_no_tf_candidate.fasta -o /path/to/list_predict_out

Outputs
--------
The program creates the following files and directories under the output directory:

```text
<project_output>/                          # Project output directory (default: output/<input_basename>/)
├── protein_model_preclassification/       # Prediction outputs (only with --predict/--list-predict)
│   ├── <name>_prediction.csv              # Predictions for all sequences
│   ├── <name>_prediction_tf_only.csv      # Subset with TF_Probability >= threshold
│   └── <name>_tf_sequences.fasta          # Extracted TF sequences (may be empty at stricter thresholds)
├── InterproScan/                          # InterProScan output directory
│   └── <input>.json
├── hmmscan/                               # hmmscan output directory
│   └── result.tbl
├── protein_kinase/                        # Protein kinase outputs (default direct/predict/--list-predict modes unless --skip-pk)
│   ├── match.json                         # Debug output (--debug): PK classification JSON in the same schema as TF/TR match.json
│   ├── <name>_pk_classified.fasta         # Classified PK sequences in the same header style as TF classified FASTA
│   ├── pk_classification.tsv              # Primary PK classification table
│   ├── shiu_classification.txt            # Shiu class summary
│   └── PPC_classification.txt             # PPC class summary
├── getrule.json                           # Debug output (--debug): parsed classification rules
└── result/                                # Classification results
    ├── match_tbl.txt                      # Final TF family classification results (table)
    ├── all_match_tbl.txt                  # Debug output (--debug): combined TF/TR + protein kinase summary table
    ├── <name>_tf_classified.fasta         # Classified TF sequences
    ├── match.json                         # Debug output (--debug)
    ├── processed_ipr_domains.json         # Domain architecture JSON (sequence ↔ domain hits; optional)
    └── pfamspec.json                      # PFAM spec JSON used for classification (optional; written with --debug)
```

Grad-CAM outputs:
- Written to `<project_output>/<name>_gradcam/` by default, with one PNG per sequence (only when --grad-cam-mode is enabled)

Workflow
--------
1. Input validation: validate FASTA structure and mixed protein/CDS inputs
2. Dependency checks: verify required tools and libraries
3. Input preprocessing: keep valid proteins and translate nucleotide entries through complete-ORF extraction
4. Prediction stage (optional): run deep learning model inference for TF preclassification
5. Protein kinase analysis: run bundled PK HMMs on the processed protein FASTA
6. Domain analysis: run InterProScan and hmmscan
7. Result processing: parse and filter analysis outputs
8. TF family classification: apply rule-based family assignment
9. Output reporting: write classification reports and FASTA files

FAQ / Troubleshooting
--------

1) db download / preparation failed (timeout / network error)
   - Download `db.tar.gz` manually, then place it in the project root (same directory as `itak` and `itak_cli.py`).
   - Re-run the program (it will try to extract / prepare db assets automatically), or extract manually:
     ```bash
     tar -xzf db.tar.gz
     ```
   - If you already have a working InterProScan, prefer using it directly:
     ```bash
     itak -i input.fasta --interproscan /path/to/interproscan.sh
     ```

2) "Dependency checks failed"
   - Ensure Python packages are installed and `java` / `hmmscan` are available on PATH.
   - Run:
     ```bash
     itak --check-deps
     ```

3) "FASTA validation failed"
   - Ensure input is valid FASTA and sequences are proteins.
   - Remove `*` (stop codon) and unexpected characters.

4) "InterProScan failed"
   - Check `java -version`, disk space, and InterProScan path (`--interproscan` if needed).

5) "Prediction is unavailable" / Grad-CAM not working
   - Install PyTorch and ensure `pre_model/model.pth` and `pre_model/predict.py` exist.
