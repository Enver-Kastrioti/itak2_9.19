================================================================================
                      iTAK2 - Transcription Factor Prediction and Analysis
================================================================================

Overview
--------
iTAK2 is a bioinformatics tool for transcription factor (TF) prediction and analysis. It integrates
deep-learning-based preclassification with conventional sequence/domain analyses to identify and
classify TFs from protein sequences.

Key Features
--------
1. TF prediction: use a deep learning model to predict whether protein sequences are potential TF/TR sequences
2. Domain analysis: run InterProScan and hmmscan for domain annotation
3. TF family classification: assign TF families based on rule-based logic
4. Output reporting: generate detailed reports and classified sequence FASTA files

System Requirements
--------
- Python 3.7+
- Java 8+ (required by InterProScan)
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
- `itak2-v1.0.py`: main CLI entry point (analysis / prediction / Grad-CAM orchestration)
- `pre_model/predict.py`: deep-learning preclassification (+ Grad-CAM heatmaps)
- `module/`: pipeline modules (FASTA processing, IPR JSON parsing, hmmscan processing, classification, checks)
- `rule.txt`: TF family classification rules

```text
itak2/
├── itak2-v1.0.py
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
│   └── interproscan/
├── hmm/
│   └── self_build.hmm
├── rule.txt
├── output/
├── temp/
└── test_protein.fasta
```

Installation
--------
## Requirements
- Python 3.7+
- Java 8+ (InterProScan)
- HMMER 3.0+ (`hmmscan`)
- PyTorch (only if you use `--predict` / `--list-predict` / `--grad-cam-mode`)

## Python Packages
```bash
pip install biopython pandas numpy
```

Optional (prediction / Grad-CAM):
```bash
# CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## InterProScan Database (db/)
- If you use the bundled InterProScan, make sure `db/` exists or `db.tar.gz` is available in the project root.
- If you use an external InterProScan via `--interproscan /path/to/interproscan.sh`, you can skip the bundled `db/`.

## Deploy / Install from GitHub
```bash
git clone https://github.com/Enver-Kastrioti/itak2_9.19.git
```

Usage
--------

Syntax:
python itak2-v1.0.py [options] -i <input_file>

Primary arguments:
-i, --input              Input FASTA file path (required except with --check-deps)
-o, --output             Output directory path (optional; default: output/<input_basename>/)
-t, --threshold          Prediction threshold in [0,1] (default: 0.1; affects prediction only)
--score                  Score threshold for InterProScan filtering (default: 1.0)
--classification-mode    Classification mode: specific (specificity-first) or score (score-first, default)

Feature options:
- -test, --test-mode      Test mode: skip InterProScan/hmmscan and validate classification using existing result files
- -json, --json-file      InterProScan JSON result file for test mode
- -spechmm, --spechmm-file hmmscan result file (result.tbl) for test mode
- --predict               Enable prediction: run model inference, then analyze predicted TF sequences
- --predict-mode          Prediction splitting mode: fast (default) or full
- --use-supplementary     Enable supplementary models (optional)
- --supplementary-only    Use supplementary models only; skip the main model
- --supp-models           Optional: specify supplementary model files (multiple allowed); if omitted, use all models in the directory
- --grad-cam-mode         Grad-CAM mode: none (default)/fast/all/positive (all/positive require --predict; --list-predict currently unsupported)
- --appl                  InterProScan application list (comma-separated; defaults to common libraries)
- --interproscan          Path to an external interproscan.sh (hmmscan will follow that InterProScan)
- --debug                 Enable debug outputs (default: off)

Dependency checks:
--check-deps            Check dependencies and exit
--skip-deps-check       Skip dependency checks (not recommended)

Examples
--------

1. Basic analysis (analyze input directly):
   python itak2-v1.0.py -i test_protein.fasta

2. Prediction workflow:
   python itak2-v1.0.py --predict -i test_protein.fasta -t 0.1

3. Specify an output directory:
   python itak2-v1.0.py --predict -i test_protein.fasta -o /path/to/output

4. Enable debug mode:
   python itak2-v1.0.py --predict -i test_protein.fasta --debug

5. Restrict InterProScan applications:
   python itak2-v1.0.py -i test_protein.fasta --appl CDD,Pfam,SMART

6. Test mode:
   python itak2-v1.0.py -test -i input.fasta -json ipr_result.json -spechmm hmmscan_result.tbl

7. Check dependencies:
   python itak2-v1.0.py --check-deps

8. Use supplementary models (requires --predict):
   python itak2-v1.0.py --predict --use-supplementary -i input.fasta
   python itak2-v1.0.py --predict --supplementary-only -i input.fasta
   python itak2-v1.0.py --predict --use-supplementary --supp-models a.pth b.pth -i input.fasta

9. Grad-CAM heatmaps (requires --predict):
   - fast: runs after classification, using the classified TF FASTA under result/ (batch mode)
   - all: runs on the original input FASTA (all sequences)
   python itak2-v1.0.py --predict --grad-cam-mode fast -i input.fasta
   python itak2-v1.0.py --predict --grad-cam-mode all -i input.fasta

Outputs
--------
The program creates the following files and directories under the output directory:

<project_output>/                         # Project output directory (default: output/<input_basename>/)
├── protein_model_preclassification/      # Prediction outputs (only with --predict/--list-predict)
│   ├── <name>_prediction.csv             # Predictions for all sequences
│   ├── <name>_prediction_tf_only.csv     # Subset with TF_Probability >= threshold
│   └── <name>_tf_sequences.fasta         # Extracted TF sequences
├── InterproScan/                         # InterProScan output directory
│   └── <input>.json
├── hmmscan/                              # hmmscan output directory
│   └── result.tbl
├── getrule.json                         # Debug output (--debug): parsed classification rules
└── result/                               # Classification results
    ├── match_tbl.txt                      # Final TF family classification results (table)
    ├── <name>_tf_classified.fasta         # Classified TF sequences
    ├── match.json                         # Debug output (--debug)
    ├── processed_ipr_domains.json         # Domain architecture JSON (sequence ↔ domain hits; optional)
    └── pfamspec.json                      # PFAM spec JSON used for classification (optional; written with --debug)

Grad-CAM outputs:
- Written to <project_output>/<name>_gradcam/ by default, with one PNG per sequence (only when --grad-cam-mode is enabled)

Workflow
--------
1. Input validation: validate FASTA structure and protein sequences
2. Dependency checks: verify required tools and libraries
3. Prediction stage (optional): run deep learning model inference for TF preclassification
4. Domain analysis: run InterProScan and hmmscan
5. Result processing: parse and filter analysis outputs
6. TF family classification: apply rule-based family assignment
7. Output reporting: write classification reports and FASTA files

FAQ / Troubleshooting
--------

1) db download / preparation failed (timeout / network error)
   - Download `db.tar.gz` manually, then place it in the project root (same directory as `itak2-v1.0.py`).
   - Re-run the program (it will try to extract / prepare db assets automatically), or extract manually:
     ```bash
     tar -xzf db.tar.gz
     ```
   - If you already have a working InterProScan, prefer using it directly:
     ```bash
     python itak2-v1.0.py -i input.fasta --interproscan /path/to/interproscan.sh
     ```

2) "Dependency checks failed"
   - Ensure Python packages are installed and `java` / `hmmscan` are available on PATH.
   - Run:
     ```bash
     python itak2-v1.0.py --check-deps
     ```

3) "FASTA validation failed"
   - Ensure input is valid FASTA and sequences are proteins.
   - Remove `*` (stop codon) and unexpected characters.

4) "InterProScan failed"
   - Check `java -version`, disk space, and InterProScan path (`--interproscan` if needed).

5) "Prediction is unavailable" / Grad-CAM not working
   - Install PyTorch and ensure `pre_model/model.pth` and `pre_model/predict.py` exist.
