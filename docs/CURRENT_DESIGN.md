# iTAK Current Design

This document is the canonical description of the current iTAK design. It supersedes earlier
transition notes and refactor plans where they disagree with the current codebase.

## Scope

iTAK currently provides:

- prediction-assisted TF/TR analysis as the default workflow
- direct TF/TR analysis as an explicit fallback
- protein kinase identification and classification
- bundled InterProScan-based domain annotation
- repository-local and managed-environment execution through a single CLI

## Meaning Of `--predict`

`--predict` should be understood as a prefilter acceleration mode, not as a second independent
analysis system.

The current intended design is:

- default invocation: first predict likely TF/TR candidates, then run the downstream domain/rule
  workflow only on the predicted candidates
- explicit fallback with `--no-predict`: run the downstream domain/rule workflow on all eligible
  sequences

So the deep-learning predictor is a front-end filter for the main analysis pipeline. Its purpose is
to reduce the amount of expensive downstream scanning, not to replace the rule-based
classification logic.

`--predict` remains accepted as a legacy-compatible spelling of the default behavior, but it no
longer changes the execution mode.

## Official Entrypoint

The only official entrypoint is:

- `itak`

The following legacy entrypoints are no longer part of the supported workflow:

- `itak3-v1.0.py`
- `run_itak3_local.sh`
- `tools/install_runtime.py`

`install_runtime.sh` remains only as a retired migration stub that tells users to use the new
runtime configuration workflow.

## Python Environment Policy

iTAK does not assume a private repository `.venv` as its primary runtime model.

Supported execution models are:

- `pixi run itak ...`
- `./itak ...` inside an existing Python / conda / bioconda environment

The `itak` launcher uses the current Python environment directly. It does not auto-switch into a
repository `.venv`.

## InterProScan Policy

iTAK now always uses the bundled InterProScan runtime under:

- `db/interproscan`

Users must not point iTAK to an external InterProScan installation.

The following are intentionally unsupported:

- `--interproscan`
- runtime configuration against an external engine directory

## Database Policy

iTAK does not use the official full InterProScan data package.

The bundled `db/interproscan` payload is project-owned and versioned by iTAK. It contains the
runtime and slimmed data required for TF/TR/PK workflows. This allows the project to evolve its
database independently of upstream InterProScan releases.

Protein kinase support is also project-owned and bundled under:

- `db/hmm_pk`

The TF/TR hmmscan database is bundled under:

- `db/hmm_self_build`

## Runtime Configuration Policy

Runtime validation and local configuration are handled by:

- `tools/configure_interproscan_runtime.py`

This tool is responsible for:

- validating the bundled InterProScan layout
- validating the bundled slimmed data layout
- activating macOS helper binaries when required
- generating `db/interproscan/interproscan.local.properties`
- running a minimal `interproscan.sh -version` self-check

The machine-readable runtime policy lives in:

- `runtime/interproscan_manifest.json`

## Helper Binary Policy

macOS support depends on project-managed helper binaries under:

- `bin/osx`

Linux helper binaries may also be shipped under:

- `bin/linux`

These helper assets are part of the iTAK runtime contract, not generic third-party runtime
discovery.

See:

- [HELPER_BINARIES.md](/Users/kentnf/projects/cornell/itak2_9.19/docs/HELPER_BINARIES.md)

## Packaging Direction

The current intended packaging direction is:

- repository workflow: `pixi`
- managed Python environments: direct `itak` invocation
- future distribution direction: `bioconda`

`pixi` is used to manage Python, Java, HMMER, and Python packages. It is not used to replace the
iTAK-managed InterProScan runtime contract.

## Testing Policy

The primary regression entrypoints are:

- `./smoke_test.sh`
- `./checkpoint_test.sh --suite quick`
- `./checkpoint_test.sh --suite full`

The runtime-specific validation entrypoints are:

- `pixi run runtime-status`
- `pixi run runtime-check`
- `./itak --check-deps`

## Document Status

This file is the current design reference.

Historical planning and migration notes have been moved to:

- [docs/archive/PIXI_RUNTIME_DESIGN.md](/Users/kentnf/projects/cornell/itak2_9.19/docs/archive/PIXI_RUNTIME_DESIGN.md)
- [docs/archive/REFACTOR_PLAN.md](/Users/kentnf/projects/cornell/itak2_9.19/docs/archive/REFACTOR_PLAN.md)
