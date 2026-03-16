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
itak2/
├── itak2-v1.0.py           # Main CLI entry point
├── module/                 # Modules
│   ├── check_dependencies.py    # Dependency checker
│   ├── validate_fasta.py        # FASTA validation
│   ├── classification.py        # TF family classification
│   ├── get_fasta.py             # Sequence extraction / FASTA utilities
│   ├── get_json.py              # InterProScan JSON processing
│   ├── get_rule.py              # Rule file parsing
│   └── selfbuild_hmm.py         # hmmscan result processing
├── pre_model/              # Prediction model assets
│   ├── model.pth               # Deep learning model file
│   └── predict.py              # Prediction script
│   └── supplementary_model/     # Supplementary models (optional)
├── db/                     # Database assets
│   ├── interproscan/           # InterProScan program and databases
├── hmm/                    # Custom HMM database (used by hmmscan)
│   └── self_build.hmm
├── Grad-Cam/               # Standalone Grad-CAM script (optional)
│   └── grad-Cam.py
├── rule.txt                # TF family classification rules
├── output/                 # Output directory
├── temp/                   # Temporary files
└── test_protein.fasta      # Example input FASTA

Installation
--------
1. Ensure Python 3.7+ is installed
2. Install required Python packages:
   pip install biopython pandas numpy

3. Install PyTorch (optional; required for prediction):
   # CPU
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # GPU (CUDA 11.8)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

4. Install HMMER:
   # Ubuntu/Debian
   sudo apt-get install hmmer
   
   # CentOS/RHEL
   sudo yum install hmmer
   
   # macOS
   brew install hmmer

5. Ensure Java is available (required by InterProScan):
   java -version

6. Database preparation (db/):
   - The program checks whether db/ is ready at runtime. If missing, it attempts to extract db.tar.gz or download and prepare the assets via module/db_manager.py.
   - If you provide an external InterProScan via --interproscan (i.e., not the iTAK2-managed InterProScan), downloading the bundled db/ assets is not required.

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
- --list-predict          Predict once, then generate multiple outputs under different thresholds
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

8. Multi-threshold batch outputs (predict once, classify multiple times):
   python itak2-v1.0.py --list-predict -i test_protein.fasta -o /path/to/output

9. Use supplementary models:
   python itak2-v1.0.py --predict --use-supplementary -i test_protein.fasta

10. Grad-CAM heatmaps (generate PNGs for classified TF sequences):
   python itak2-v1.0.py --predict --grad-cam-mode fast -i test_protein.fasta

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
    ├── match_tbl.txt                      # TF family assignment table
    ├── <name>_tf_classified.fasta         # Classified TF sequences
    ├── match.json                         # Debug output (--debug)
    ├── processed_ipr_domains.json         # Debug output
    └── pfamspec.json                      # Debug output (--debug)

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

Troubleshooting
--------

Common issues:

1. "Dependency checks failed"
   - Verify required Python packages are installed
   - Confirm external tools are available on PATH
   - Run --check-deps for detailed diagnostics

2. "FASTA validation failed"
   - Confirm the FASTA file format is valid
   - Verify sequences are protein sequences
   - Remove asterisks (*) from sequences

3. "InterProScan failed"
   - Verify Java is installed and accessible
   - Check integrity of InterProScan databases
   - Ensure sufficient disk space

4. "Prediction is unavailable"
   - Install PyTorch: pip install torch
   - Confirm model.pth exists
   - Verify predict.py is present and intact
