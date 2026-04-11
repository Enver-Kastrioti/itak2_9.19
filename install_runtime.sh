#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cat >&2 <<EOF
install_runtime.sh has been retired.

Use one of the explicit runtime setup commands instead:
  pixi run configure-runtime
  pixi run runtime-status
  pixi run runtime-check

If you are not using pixi, run:
  python3 "$ROOT_DIR/tools/configure_interproscan_runtime.py"
EOF
exit 1
