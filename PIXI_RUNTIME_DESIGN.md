# iTAK Runtime And Packaging Design

## Goal

Define a runtime model that supports all of the following without ambiguity:

- repository-local development and testing
- future `pixi`-managed environments
- future `bioconda` packaging
- bundled slimmed InterProScan data that remains under iTAK control
- macOS-specific helper binaries required to make InterProScan runnable


## Non-Negotiable Constraints

### 1. InterProScan data must remain iTAK-managed

iTAK must not switch to the official full InterProScan data package.

Reason:

- the current `db/` layout is a slimmed database prepared specifically for TF/TR/PK workflows
- this bundled data will continue to evolve independently of upstream InterProScan releases
- output behavior and runtime expectations depend on this reduced dataset

Implication:

- `data.directory` must always point to the iTAK-managed database
- external or system-provided InterProScan installations may provide the engine only, not the data payload used by iTAK


### 2. macOS requires iTAK-managed helper binaries

macOS support depends on helper executables compiled and shipped by this project.

Reason:

- upstream or system-provided InterProScan may not be runnable on macOS as-is
- specific subprograms must be replaced or redirected to iTAK-provided binaries for successful execution

Implication:

- macOS support is not just "use any InterProScan"
- iTAK must manage a helper-binary activation layer on macOS
- engine compatibility on macOS must be treated as a supported-version matrix, not as an open-ended promise


## Runtime Model

InterProScan support is split into three separate layers.

### Engine

The InterProScan program itself:

- `interproscan.sh`
- jars
- engine-side support files

This layer may come from:

- Linux: `conda`, `bioconda`, `pixi`, or a user-provided installation
- macOS: user-provided installation, plus iTAK helper activation


### Data

The InterProScan data used by iTAK.

This layer must come only from iTAK-managed assets:

- current repository `db/`
- future versioned iTAK data bundles


### Helpers

Platform-specific native executables required to make the engine usable.

This layer is managed by iTAK:

- Linux: may be unnecessary when upstream binaries are compatible
- macOS: required, and must come from iTAK


## Platform Strategy

### Linux

Preferred model:

- engine from `pixi` / `conda` / `bioconda`
- data from iTAK-managed `db`
- helper overrides optional, only if needed

Operationally:

- detect engine path
- generate `interproscan.local.properties`
- force `data.directory` to iTAK data
- run with system or environment-provided Java/HMMER as declared by the environment


### macOS

Preferred model:

- engine from a user-provided InterProScan installation
- data from iTAK-managed `db`
- helper binaries from iTAK-managed `bin/osx`

Operationally:

- detect or require a user-supplied `interproscan.sh`
- validate engine version against an explicit supported list
- activate iTAK macOS helper binaries into the chosen engine layout
- generate `interproscan.local.properties`
- force `data.directory` to iTAK data

Important:

- macOS support should be version-scoped
- iTAK should support only tested engine versions
- unsupported engine versions should produce a clear compatibility warning or a hard failure


## Versioning Model

The project should track three version surfaces independently.

- `engine_version`
- `db_version`
- `helper_version`

These versions should not be conflated into a single "runtime version".


## Proposed Manifest

Add a machine-readable manifest, for example:

- `runtime/interproscan_manifest.json`

Recommended fields:

```json
{
  "db_version": "2026.04",
  "data_layout_version": 1,
  "helper_version": "2026.04-macos",
  "engine_policy": {
    "linux": {
      "supported": ["5.72-103.0", "5.75-106.0"]
    },
    "macos": {
      "supported": ["5.75-106.0"],
      "requires_itak_helpers": true
    }
  }
}
```

This manifest should drive:

- dependency validation
- install/configure decisions
- compatibility warnings
- future db/helper upgrades


## Role Of pixi

`pixi` should manage the software environment, not the InterProScan data contract.

### pixi should manage

- `python`
- `java`
- `hmmer`
- Python package dependencies
- Linux-side engine installation, when available and appropriate


### pixi should not manage

- iTAK slimmed InterProScan data versioning
- macOS helper binaries
- iTAK runtime compatibility policy


## Role Of bioconda

`bioconda` should eventually become the packaging and distribution channel for the application, but not for iTAK-specific data evolution policy.

Preferred packaging posture:

- install CLI entrypoint `itak`
- install Python dependencies
- install Java/HMMER/possibly Linux engine dependencies
- keep iTAK data and helper compatibility under project control

The future `bioconda` package should not:

- create a private `.venv`
- run `pip install` at runtime
- silently fetch official full InterProScan data in place of iTAK-managed data


## Status Of install_runtime.sh

`install_runtime.sh` is now retired.

### Current state

It exits with migration guidance and points users to:

- `pixi run configure-runtime`
- `pixi run runtime-status`
- `pixi run runtime-check`
- `python3 tools/configure_interproscan_runtime.py`


### Target state

The replacement workflow should primarily:

- validate engine/data/helper compatibility
- generate local runtime configuration
- activate macOS helper binaries when required
- verify the final runtime with a lightweight self-check

Long-term outcome:

- keep explicit runtime configuration in Python and/or `pixi` tasks rather than restoring a shell bootstrap installer


## Future Of .venv

`.venv` is acceptable as a repository-local convenience for today, but it should not be the long-term core design.

Long-term direction:

- repository development: `pixi`
- packaged distribution: `bioconda`
- runtime entrypoint: `itak`

The `itak` launcher should eventually stop assuming `.venv` is the preferred global answer once `pixi` becomes the primary source workflow.


## Proposed Transition Plan

### Phase 1: Freeze runtime semantics

- keep `itak` as the only official entrypoint
- keep iTAK-managed `db`
- keep iTAK-managed macOS helper binaries
- introduce the runtime manifest


### Phase 2: Add pixi support

- create `pixi.toml`
- declare Python, Java, HMMER, and core Python packages
- add runnable tasks such as:
  - `pixi run itak --help`
  - `pixi run smoke-test`
  - `pixi run checkpoint-quick`


### Phase 3: Separate engine configuration from environment bootstrap

- keep runtime configuration in `tools/configure_interproscan_runtime.py`
- reduce reliance on `.venv`
- keep a compatibility path for repository users who do not use `pixi`


### Phase 4: Prepare bioconda packaging

- move dependency declarations into package metadata
- ensure `itak` runs correctly inside a managed external environment
- keep db/helper policy under iTAK control


## Recommended Immediate Next Actions

1. Add `runtime/interproscan_manifest.json`.
2. Refactor dependency checks to read compatibility policy from the manifest instead of hardcoding assumptions.
3. Draft `pixi.toml` with Python, Java, HMMER, and Python package dependencies only.
4. Keep InterProScan data and macOS helper management outside `pixi`.
5. Decide whether macOS should support:
   - only a pinned engine version
   - or a small tested version range


## Decision Summary

- iTAK data stays project-owned and versioned by iTAK.
- macOS helper binaries stay project-owned and versioned by iTAK.
- Linux may use external engine packages, but not external data.
- macOS may use external engine installations, but only with iTAK helper activation and version checks.
- `pixi` is the right direction for repository environment management.
- `bioconda` remains the right long-term distribution target.
