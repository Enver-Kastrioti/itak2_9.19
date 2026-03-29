#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE="quick"
LABEL=""
OUTPUT_ROOT=""
APPL_LIST="PROSITEPROFILES"

usage() {
  cat <<'EOF'
Usage: ./checkpoint_test.sh [options]

Options:
  --suite NAME       quick (default) or full
  --label NAME       Label appended to output directory names
  --output-root DIR  Base directory for generated test outputs
  --appl LIST        InterProScan application list (default: PROSITEPROFILES)
  --help             Show this help message

Suites:
  quick              Syntax + direct baseline + positive PK direct regression
  full               quick + positive PK predict regression + positive PK debug regression

Examples:
  ./checkpoint_test.sh
  ./checkpoint_test.sh --suite full
  ./checkpoint_test.sh --suite full --label cp1_context
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --appl)
      APPL_LIST="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "$SUITE" != "quick" && "$SUITE" != "full" ]]; then
  echo "Unsupported suite: $SUITE" >&2
  exit 1
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
  if [[ -n "$LABEL" ]]; then
    OUTPUT_ROOT="$ROOT_DIR/output/checkpoints/$LABEL"
  else
    OUTPUT_ROOT="$ROOT_DIR/output/checkpoints/$SUITE"
  fi
fi

mkdir -p "$OUTPUT_ROOT"

run_step() {
  local name="$1"
  shift
  echo
  echo "== $name =="
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

check_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Expected file was not created: $path" >&2
    exit 1
  fi
}

run_step "Syntax Check" python -m py_compile \
  "$ROOT_DIR/itak3-v1.0.py" \
  "$ROOT_DIR/module/check_dependencies.py" \
  "$ROOT_DIR/module/protein_kinase.py" \
  "$ROOT_DIR/module/output_contracts.py" \
  "$ROOT_DIR/module/runtime_tools.py"

run_step "Direct Baseline" \
  "$ROOT_DIR/smoke_test.sh" \
  --input "$ROOT_DIR/test_protein.fasta" \
  --appl "$APPL_LIST" \
  --output "$OUTPUT_ROOT/direct_default"

run_step "Direct PK Positive" \
  "$ROOT_DIR/smoke_test.sh" \
  --input "$ROOT_DIR/test_protein_kinase.fasta" \
  --appl "$APPL_LIST" \
  --require-pk 2 \
  --output "$OUTPUT_ROOT/direct_pk_positive"

if [[ "$SUITE" == "full" ]]; then
  run_step "Predict PK Positive" \
    "$ROOT_DIR/smoke_test.sh" \
    --predict \
    --input "$ROOT_DIR/test_protein_kinase.fasta" \
    --appl "$APPL_LIST" \
    --require-pk 2 \
    --output "$OUTPUT_ROOT/predict_pk_positive"

  run_step "Predict PK Positive Without TF" \
    "$ROOT_DIR/run_itak3_local.sh" \
    --predict \
    -t 0.3 \
    -i "$ROOT_DIR/test_pk_no_tf_candidate.fasta" \
    --appl "$APPL_LIST" \
    -o "$OUTPUT_ROOT/predict_pk_no_tf"

  check_file "$OUTPUT_ROOT/predict_pk_no_tf/result/match_tbl.txt"
  check_file "$OUTPUT_ROOT/predict_pk_no_tf/result/all_match_tbl.txt"
  check_file "$OUTPUT_ROOT/predict_pk_no_tf/protein_kinase/pk_classification.tsv"

  if [[ -s "$OUTPUT_ROOT/predict_pk_no_tf/result/match_tbl.txt" ]]; then
    echo "Expected empty TF/TR match table for no-TF prediction case" >&2
    exit 1
  fi

  no_tf_pk_count="$(tail -n +2 "$OUTPUT_ROOT/predict_pk_no_tf/protein_kinase/pk_classification.tsv" | awk 'NF{count++} END{print count+0}')"
  if [[ "$no_tf_pk_count" -lt 1 ]]; then
    echo "Expected at least 1 protein kinase classification in no-TF prediction case" >&2
    exit 1
  fi

  no_tf_summary_count="$(tail -n +2 "$OUTPUT_ROOT/predict_pk_no_tf/result/all_match_tbl.txt" | awk 'NF{count++} END{print count+0}')"
  if [[ "$no_tf_summary_count" -lt 1 ]]; then
    echo "Expected combined summary rows for no-TF prediction case" >&2
    exit 1
  fi

  run_step "Debug PK Positive" \
    "$ROOT_DIR/run_itak3_local.sh" \
    -i "$ROOT_DIR/test_protein_kinase.fasta" \
    --appl "$APPL_LIST" \
    --debug \
    -o "$OUTPUT_ROOT/debug_pk_positive"

  check_file "$OUTPUT_ROOT/debug_pk_positive/result/match.json"
  check_file "$OUTPUT_ROOT/debug_pk_positive/protein_kinase/match.json"
fi

echo
echo "Checkpoint test suite passed: $SUITE"
echo "Output root: $OUTPUT_ROOT"
