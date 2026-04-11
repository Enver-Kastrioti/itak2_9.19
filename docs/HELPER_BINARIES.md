# InterProScan Helper Binaries

This file was moved from `bin/SOURCES.md` so project-level documentation can live under `docs/`.

This repository now contains a split helper-binary layout under `bin/`:

- `bin/linux/`: binaries and scripts copied from the bundled `db/interproscan`
- `bin/osx/`: macOS-compatible replacements gathered from local `pixi` packages or built from source

## Linux

These files were copied from the repository's bundled InterProScan payload:

- `bin/linux/cdd/`: `rpsblast`, `rpsbproc`
- `bin/linux/hmmer2/`: `hmmpfam`
- `bin/linux/hmmer3/`: `hmmsearch`, `hmmscan`, `hmmpress`, `esl-translate`
- `bin/linux/panther/`: `epa-ng.ubuntu20.04`, `treegrafter.py`
- `bin/linux/prosite/`: `ps_scan.pl`, `runprosite.py`, `pfsearch_wrapper.py`, `psa2msa`, `pfscan`, `pfsearch`, `pfscanV3`, `pfsearchV3`

## macOS

### Pulled from local `pixi`

The following native Apple Silicon binaries were copied from `.pixi-bin-osx/.pixi/envs/default`:

- `bin/osx/cdd/`: `rpsblast`, `rpsbproc`
- `bin/osx/hmmer2/`: `hmmpfam`
- `bin/osx/hmmer3/`: `hmmsearch`, `hmmscan`, `hmmpress`, `esl-translate`
- `bin/osx/panther/`: `epa-ng`

The following helper scripts were copied from the bundled InterProScan tree because they are portable scripts:

- `bin/osx/panther/treegrafter.py`
- `bin/osx/prosite/ps_scan.pl`
- `bin/osx/prosite/runprosite.py`
- `bin/osx/prosite/pfsearch_wrapper.py`
- `bin/osx/prosite/psa2msa`

Runtime shared libraries needed by the macOS binaries were copied into `bin/osx/lib/`.

### Built from source for macOS

`pfscanV3` and `pfsearchV3` were built from source from:

- upstream: `https://github.com/sib-swiss/pftools3`
- local checkout: `.build/pftools3`

Build notes:

- built with CMake in `.build/pftools3/build-osx-x86_64`
- built as `x86_64` Mach-O binaries for macOS
- intended to run on Apple Silicon via Rosetta, or directly on Intel Macs
- native `arm64` build was not completed because upstream `pftools3` is tightly coupled to SSE code paths and Linux-oriented assumptions

Installed outputs:

- `bin/osx/prosite/pfscanV3`
- `bin/osx/prosite/pfsearchV3`

## Verification

The following spot checks were run successfully on this machine:

- `bin/osx/prosite/pfscanV3`
- `bin/osx/prosite/pfsearchV3`
- `bin/osx/hmmer3/hmmscan -h`
- `bin/osx/cdd/rpsblast -version`
- `bin/osx/panther/epa-ng --help`

Note:

- `ps_scan.pl` still defaults to `pfscan` unless a `pfscanV3` path is passed explicitly.
