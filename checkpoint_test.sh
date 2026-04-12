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
  quick              Syntax + default predictive baseline + direct fallback PK regression
  full               quick + predictive PK regression + list-predict + debug regression

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

ITAK_CMD=()
if [[ -x "$ROOT_DIR/itak" ]]; then
  ITAK_CMD=("$ROOT_DIR/itak")
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  ITAK_CMD=("$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/itak")
else
  ITAK_CMD=("python3" "$ROOT_DIR/itak")
fi

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

extract_fasta_subset() {
  local input_fasta="$1"
  local output_fasta="$2"
  shift 2

  awk -v ids="$(printf '%s\n' "$@")" '
    BEGIN {
      split(ids, raw_ids, "\n")
      for (i in raw_ids) {
        if (raw_ids[i] != "") {
          keep[raw_ids[i]] = 1
        }
      }
    }
    /^>/ {
      header = substr($0, 2)
      emit = (header in keep)
    }
    emit {
      print
    }
  ' "$input_fasta" > "$output_fasta"

  if [[ ! -s "$output_fasta" ]]; then
    echo "Failed to generate FASTA subset: $output_fasta" >&2
    exit 1
  fi
}

check_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Expected file was not created: $path" >&2
    exit 1
  fi
}

run_step "Syntax Check" python -m py_compile \
  "$ROOT_DIR/itak" \
  "$ROOT_DIR/itak_cli.py" \
  "$ROOT_DIR/module/check_dependencies.py" \
  "$ROOT_DIR/module/protein_kinase.py" \
  "$ROOT_DIR/module/output_contracts.py" \
  "$ROOT_DIR/module/runtime_tools.py"

run_step "Default Predict Baseline" \
  "$ROOT_DIR/smoke_test.sh" \
  --input "$ROOT_DIR/test_protein.fasta" \
  --appl "$APPL_LIST" \
  --require-pk 2 \
  --output "$OUTPUT_ROOT/default_predict"

run_step "Direct Fallback PK Positive" \
  "$ROOT_DIR/smoke_test.sh" \
  --no-predict \
  --input "$ROOT_DIR/test_protein.fasta" \
  --appl "$APPL_LIST" \
  --require-pk 2 \
  --output "$OUTPUT_ROOT/direct_no_predict_pk_positive"

if [[ "$SUITE" == "full" ]]; then
  FIXTURE_DIR="$OUTPUT_ROOT/generated_fixtures"
  mkdir -p "$FIXTURE_DIR"

  PK_NO_TF_FASTA="$FIXTURE_DIR/test_pk_no_tf_candidate.fasta"
  extract_fasta_subset \
    "$ROOT_DIR/test_protein.fasta" \
    "$PK_NO_TF_FASTA" \
    "AT1G01450.1"

  run_step "Default Predict PK Positive" \
    "$ROOT_DIR/smoke_test.sh" \
    --input "$ROOT_DIR/test_protein.fasta" \
    --appl "$APPL_LIST" \
    --require-pk 2 \
    --output "$OUTPUT_ROOT/predict_pk_positive"

  run_step "Predict PK Positive Without TF" \
    "${ITAK_CMD[@]}" \
    -t 0.3 \
    -i "$PK_NO_TF_FASTA" \
    --appl "$APPL_LIST" \
    -o "$OUTPUT_ROOT/predict_pk_no_tf"

  check_file "$OUTPUT_ROOT/predict_pk_no_tf/result/match_tbl.txt"
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

  run_step "List Predict PK Contract" \
    "${ITAK_CMD[@]}" \
    --list-predict \
    -i "$PK_NO_TF_FASTA" \
    --appl "$APPL_LIST" \
    -o "$OUTPUT_ROOT/list_predict_contract"

  LIST_BASE="$OUTPUT_ROOT/list_predict_contract"
  LIST_PREFIX="test_pk_no_tf_candidate"
  check_file "$LIST_BASE/${LIST_PREFIX}_no_pre/protein_kinase/pk_classification.tsv"
  check_file "$LIST_BASE/${LIST_PREFIX}_10/result/match_tbl.txt"
  check_file "$LIST_BASE/${LIST_PREFIX}_10/protein_kinase/pk_classification.tsv"
  check_file "$LIST_BASE/${LIST_PREFIX}_10/protein_model_preclassification/${LIST_PREFIX}_prediction.csv"
  check_file "$LIST_BASE/${LIST_PREFIX}_10/protein_model_preclassification/${LIST_PREFIX}_protein_replaced_tf_sequences.fasta"
  check_file "$LIST_BASE/${LIST_PREFIX}_30/result/match_tbl.txt"
  check_file "$LIST_BASE/${LIST_PREFIX}_30/protein_kinase/pk_classification.tsv"

  if [[ ! -s "$LIST_BASE/${LIST_PREFIX}_10/protein_model_preclassification/${LIST_PREFIX}_protein_replaced_tf_sequences.fasta" ]]; then
    echo "Expected non-empty TF FASTA for --list-predict threshold 0.1" >&2
    exit 1
  fi

  if [[ -s "$LIST_BASE/${LIST_PREFIX}_30/result/match_tbl.txt" ]]; then
    echo "Expected empty TF/TR match table for --list-predict threshold 0.3" >&2
    exit 1
  fi

  list_pk_count="$(tail -n +2 "$LIST_BASE/${LIST_PREFIX}_30/protein_kinase/pk_classification.tsv" | awk 'NF{count++} END{print count+0}')"
  if [[ "$list_pk_count" -lt 1 ]]; then
    echo "Expected at least 1 protein kinase classification for --list-predict threshold 0.3" >&2
    exit 1
  fi

  run_step "Debug PK Positive" \
    "${ITAK_CMD[@]}" \
    -i "$ROOT_DIR/test_protein.fasta" \
    --appl "$APPL_LIST" \
    --debug \
    -o "$OUTPUT_ROOT/debug_pk_positive"

  check_file "$OUTPUT_ROOT/debug_pk_positive/result/match.json"
  check_file "$OUTPUT_ROOT/debug_pk_positive/protein_kinase/match.json"
fi

echo
echo "Checkpoint test suite passed: $SUITE"
echo "Output root: $OUTPUT_ROOT"
