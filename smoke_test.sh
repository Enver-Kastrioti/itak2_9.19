#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_FASTA="$ROOT_DIR/test_protein.fasta"
OUTPUT_DIR="$ROOT_DIR/output/smoke_test"
APPL_LIST="PROSITEPROFILES"
RUN_PREDICT=1
REQUIRE_PK_COUNT=""
CPU_COUNT=""

usage() {
  cat <<'EOF'
Usage: ./smoke_test.sh [options]

Options:
  --no-predict        Disable the default predictive prefilter and analyze all sequences directly
  --input PATH        Input FASTA file (default: test_protein.fasta)
  --output PATH       Output directory (default: output/smoke_test)
  --cpu N             CPU threads forwarded to itak
  --appl LIST         InterProScan applications (default: PROSITEPROFILES)
  --require-pk N      Require at least N protein kinase classifications
  --help              Show this help message

Examples:
  ./smoke_test.sh
  ./smoke_test.sh --no-predict
  ./smoke_test.sh --cpu 8
  ./smoke_test.sh --require-pk 2
  ./smoke_test.sh --no-predict --require-pk 2
  ./smoke_test.sh --appl CDD,Pfam,SMART
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-predict)
      RUN_PREDICT=0
      shift
      ;;
    --input)
      INPUT_FASTA="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --cpu)
      CPU_COUNT="$2"
      shift 2
      ;;
    --appl)
      APPL_LIST="$2"
      shift 2
      ;;
    --require-pk)
      REQUIRE_PK_COUNT="$2"
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

if [[ ! -f "$INPUT_FASTA" ]]; then
  echo "Input FASTA not found: $INPUT_FASTA" >&2
  exit 1
fi

CMD=()
if [[ -x "$ROOT_DIR/itak" ]]; then
  CMD=("$ROOT_DIR/itak")
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  CMD=("$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/itak")
else
  CMD=("python3" "$ROOT_DIR/itak")
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

ARGS=(-i "$INPUT_FASTA" --appl "$APPL_LIST" -o "$OUTPUT_DIR")
if [[ -n "$CPU_COUNT" ]]; then
  ARGS=(--cpu "$CPU_COUNT" "${ARGS[@]}")
fi
if [[ "$RUN_PREDICT" -eq 0 ]]; then
  ARGS=(--no-predict "${ARGS[@]}")
fi

echo "Running smoke test..."
printf '+'
printf ' %q' "${CMD[@]}" "${ARGS[@]}"
printf '\n'
"${CMD[@]}" "${ARGS[@]}"

check_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Expected output was not created: $path" >&2
    exit 1
  fi
}

json_file="$(find "$OUTPUT_DIR/InterproScan" -maxdepth 1 -name '*.json' -print -quit 2>/dev/null || true)"
check_file "$json_file"
check_file "$OUTPUT_DIR/hmmscan/result.tbl"
check_file "$OUTPUT_DIR/result/match_tbl.txt"
check_file "$OUTPUT_DIR/protein_kinase/pk_classification.tsv"
pk_classified_fasta="$(find "$OUTPUT_DIR/protein_kinase" -maxdepth 1 -name '*_pk_classified.fasta' -print -quit 2>/dev/null || true)"
check_file "$pk_classified_fasta"
check_file "$OUTPUT_DIR/protein_kinase/shiu_classification.txt"
check_file "$OUTPUT_DIR/protein_kinase/PPC_classification.txt"

pk_count="$(tail -n +2 "$OUTPUT_DIR/protein_kinase/pk_classification.tsv" | awk 'NF{count++} END{print count+0}')"
if [[ -n "$REQUIRE_PK_COUNT" ]]; then
  if [[ "$pk_count" -lt "$REQUIRE_PK_COUNT" ]]; then
    echo "Expected at least $REQUIRE_PK_COUNT protein kinase classifications, found $pk_count" >&2
    exit 1
  fi
fi

if [[ "$RUN_PREDICT" -eq 1 ]]; then
  check_file "$OUTPUT_DIR/protein_model_preclassification/$(basename "${INPUT_FASTA%.*}")_prediction.csv"
  check_file "$OUTPUT_DIR/protein_model_preclassification/$(basename "${INPUT_FASTA%.*}")_prediction_tf_only.csv"
fi

echo "Smoke test passed."
echo "Output directory: $OUTPUT_DIR"
echo "InterProScan JSON: $json_file"
echo "Classification table: $OUTPUT_DIR/result/match_tbl.txt"
echo "Protein kinase FASTA: $pk_classified_fasta"
echo "Protein kinase classifications: $pk_count"
