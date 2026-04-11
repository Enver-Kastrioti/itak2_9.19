#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import textwrap
import venv
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from module.runtime_tools import (  # noqa: E402
    activate_bundled_interproscan_binaries,
    build_runtime_env,
    get_helper_root,
    resolve_java_executable,
)
from module.db_manager import setup_db, validate_interproscan_installation  # noqa: E402


DEFAULT_INTERPROSCAN_DIR = ROOT_DIR / "db" / "interproscan"


def shell_join(parts):
    return " ".join(shlex.quote(str(part)) for part in parts)


def run_command(cmd, *, env=None, cwd=None):
    print(f"+ {shell_join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=cwd or ROOT_DIR)


def create_virtualenv(python_bin, venv_dir):
    if not venv_dir.exists():
        print(f"Creating virtual environment: {venv_dir}")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(str(venv_dir))

    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def detect_perl():
    perl_path = shutil.which("perl")
    if perl_path:
        return perl_path
    return "perl"


def update_property_line(line, replacements):
    for key, value in replacements.items():
        prefix = f"{key}="
        if line.startswith(prefix):
            return prefix + value
    return line


def load_properties(path):
    properties = {}
    if not path.exists():
        return properties

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        properties[key.strip()] = value.strip()
    return properties


def is_relative_to(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def write_local_interproscan_config(template_path, output_path, *, bin_dir, data_dir, python_cmd, perl_cmd):
    replacements = {
        "bin.directory": str(bin_dir),
        "data.directory": str(data_dir),
        "python3.command": str(python_cmd),
        "perl.command": str(perl_cmd),
    }

    lines = template_path.read_text(encoding="utf-8").splitlines()
    rendered = [update_property_line(line, replacements) for line in lines]
    output_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def build_runtime_shell_lines(*, java_bin, helper_root, config_path):
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "",
        f'CONFIG_PATH="{config_path}"',
        'if [[ ! -f "$CONFIG_PATH" ]]; then',
        '  echo "Runtime configuration not found: $CONFIG_PATH" >&2',
        '  echo "Run ./install_runtime.sh first to generate local runtime files." >&2',
        '  exit 1',
        'fi',
    ]

    if java_bin:
        lines.append(f'export PATH="{java_bin.parent}:$PATH"')

    if helper_root:
        helper_lib = helper_root / "lib"
        if helper_lib.exists():
            if sys.platform == "darwin":
                lines.append(f'export DYLD_LIBRARY_PATH="{helper_lib}:${{DYLD_LIBRARY_PATH:-}}"')
            else:
                lines.append(f'export LD_LIBRARY_PATH="{helper_lib}:${{LD_LIBRARY_PATH:-}}"')

    lines.append('export INTERPROSCAN_CONF="$CONFIG_PATH"')
    return lines


def write_interproscan_wrapper(wrapper_path, *, java_bin, helper_root, config_path, interproscan_script):
    lines = build_runtime_shell_lines(java_bin=java_bin, helper_root=helper_root, config_path=config_path)
    lines.extend(
        [
            f'exec "{interproscan_script}" "$@"',
            "",
        ]
    )
    wrapper_path.write_text("\n".join(lines), encoding="utf-8")
    wrapper_path.chmod(0o755)


def write_itak_wrapper(wrapper_path, *, java_bin, helper_root, config_path, python_bin, interproscan_script=None):
    lines = build_runtime_shell_lines(java_bin=java_bin, helper_root=helper_root, config_path=config_path)
    itak_exec = f'exec "{python_bin}" "$SCRIPT_DIR/itak"'
    if interproscan_script is not None:
        itak_exec += f' --interproscan "{interproscan_script}"'
    lines.extend(
        [
            f'{itak_exec} "$@"',
            "",
        ]
    )
    wrapper_path.write_text("\n".join(lines), encoding="utf-8")
    wrapper_path.chmod(0o755)


def install_python_packages(python_bin, *, with_predict, predict_backend, torch_index_url):
    run_command([str(python_bin), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run_command([str(python_bin), "-m", "pip", "install", "-r", str(ROOT_DIR / "requirements-core.txt")])
    if with_predict:
        cmd = [str(python_bin), "-m", "pip", "install", "-r", str(ROOT_DIR / "requirements-predict.txt")]
        if predict_backend in {"cpu", "cuda"} and torch_index_url:
            cmd.extend(["--index-url", torch_index_url])
        run_command(cmd)


def run_dependency_check(python_bin, *, config_path, interproscan_script=None):
    env = build_runtime_env(ROOT_DIR)
    env["INTERPROSCAN_CONF"] = str(config_path)
    cmd = [str(python_bin), str(ROOT_DIR / "itak"), "--check-deps"]
    if interproscan_script is not None:
        cmd.extend(["--interproscan", str(interproscan_script)])
    run_command(cmd, env=env)


def run_smoke_test(
    python_bin,
    *,
    wrapper_path,
    config_path,
    input_fasta,
    appl_list,
    output_root,
    interproscan_script=None,
):
    input_fasta = Path(input_fasta).resolve()
    if not input_fasta.exists():
        raise SystemExit(f"Smoke test input FASTA not found: {input_fasta}")

    env = build_runtime_env(ROOT_DIR)
    env["INTERPROSCAN_CONF"] = str(config_path)

    interproscan_out = output_root / "interproscan"
    itak_out = output_root / "itak"
    interproscan_out.mkdir(parents=True, exist_ok=True)
    itak_out.mkdir(parents=True, exist_ok=True)

    print(f"Running smoke test with input: {input_fasta}")
    run_command(
        [
            str(wrapper_path),
            "-i",
            str(input_fasta),
            "-f",
            "json",
            "-d",
            str(interproscan_out),
            "-appl",
            appl_list,
        ],
        env=env,
    )
    run_command(
        [
            str(python_bin),
            str(ROOT_DIR / "itak"),
            "-i",
            str(input_fasta),
            "--appl",
            appl_list,
            "--output",
            str(itak_out),
            *(["--interproscan", str(interproscan_script)] if interproscan_script is not None else []),
        ],
        env=env,
    )


def clean_runtime(*, interproscan_dir, venv_dir, remove_venv):
    removed = []
    candidates = [
        ROOT_DIR / "run_interproscan_local.sh",
        ROOT_DIR / "run_itak2_local.sh",
        ROOT_DIR / "run_itak3_local.sh",
        interproscan_dir / "interproscan.local.properties",
    ]

    for path in candidates:
        if path.exists() or path.is_symlink():
            path.unlink()
            removed.append(path)

    if remove_venv and venv_dir.exists():
        if is_relative_to(venv_dir, ROOT_DIR):
            shutil.rmtree(venv_dir)
            removed.append(venv_dir)
        else:
            raise SystemExit(f"Refusing to remove virtualenv outside repository: {venv_dir}")

    return removed


def print_interproscan_issues(issues):
    for issue in issues:
        print(f"    - {issue}")


def confirm_bundled_db_repair(message, *, auto_download):
    print(message)
    if auto_download:
        print("Auto-confirming bundled db preparation because --download-db was specified.")
        return True

    if not sys.stdin.isatty():
        print("Cannot prompt in non-interactive mode.")
        print("Re-run with --download-db to allow automatic bundled db preparation.")
        return False

    reply = input("Download and extract the bundled db/interproscan now? [y/N]: ").strip().lower()
    return reply in {"y", "yes"}


def ensure_usable_interproscan(interproscan_dir, *, auto_download):
    interproscan_dir = Path(interproscan_dir).resolve()
    default_dir = DEFAULT_INTERPROSCAN_DIR.resolve()
    requested_is_default = interproscan_dir == default_dir

    issues = validate_interproscan_installation(interproscan_dir)
    if not issues:
        return interproscan_dir

    print(f"InterProScan installation check failed for: {interproscan_dir}")
    print_interproscan_issues(issues)

    if requested_is_default:
        prompt_message = (
            "The bundled InterProScan under db/interproscan is missing or incomplete.\n"
            "iTAK can remove the broken bundled copy and prepare a fresh one from db.tar.gz."
        )
    else:
        prompt_message = (
            f"The specified --interproscan-dir is not usable: {interproscan_dir}\n"
            f"iTAK can instead prepare the bundled copy under: {default_dir}"
        )

    if not confirm_bundled_db_repair(prompt_message, auto_download=auto_download):
        print(
            "A valid InterProScan installation is required. "
            "Provide a working --interproscan-dir or re-run with --download-db."
        )
        raise SystemExit(1)

    if not setup_db(ROOT_DIR):
        print("Failed to prepare bundled db/interproscan from db.tar.gz.")
        raise SystemExit(1)

    repaired_issues = validate_interproscan_installation(default_dir)
    if repaired_issues:
        print("Bundled InterProScan is still incomplete after attempted repair:")
        print_interproscan_issues(repaired_issues)
        print("Bundled InterProScan repair failed.")
        raise SystemExit(1)

    print(f"Using bundled InterProScan: {default_dir}")
    return default_dir


def print_status(*, interproscan_dir, venv_dir):
    local_config = interproscan_dir / "interproscan.local.properties"
    interproscan_wrapper = ROOT_DIR / "run_interproscan_local.sh"
    legacy_itak_wrapper = ROOT_DIR / "run_itak3_local.sh"
    itak_entry = ROOT_DIR / "itak"
    properties = load_properties(local_config)
    interproscan_issues = validate_interproscan_installation(interproscan_dir)
    interproscan_ok = not interproscan_issues

    installed = (
        local_config.exists()
        and interproscan_wrapper.exists()
        and itak_entry.exists()
        and interproscan_ok
    )
    print("Runtime status:")
    print(f"  Installed: {'yes' if installed else 'no'}")
    print(f"  InterProScan dir: {interproscan_dir} ({'valid' if interproscan_ok else 'invalid'})")
    print(f"  Local config: {local_config} ({'present' if local_config.exists() else 'missing'})")
    print(f"  InterProScan wrapper: {interproscan_wrapper} ({'present' if interproscan_wrapper.exists() else 'missing'})")
    print(f"  iTAK entry script: {itak_entry} ({'present' if itak_entry.exists() else 'missing'})")
    print(
        f"  Deprecated compatibility wrapper: {legacy_itak_wrapper} "
        f"({'present' if legacy_itak_wrapper.exists() else 'missing'})"
    )
    print(f"  Selected venv path: {venv_dir} ({'present' if venv_dir.exists() else 'missing'})")
    print("  Preferred CLI: ./itak")
    print("  Compatibility note: do not script against run_itak3_local.sh in new setups")
    if interproscan_issues:
        print("  InterProScan issues:")
        print_interproscan_issues(interproscan_issues)

    runtime_python = properties.get("python3.command")
    runtime_perl = properties.get("perl.command")
    runtime_bin = properties.get("bin.directory")
    runtime_data = properties.get("data.directory")

    print(f"  Python: {runtime_python or 'unknown'}")
    print(f"  Perl: {runtime_perl or 'unknown'}")
    print(f"  InterProScan bin dir: {runtime_bin or 'unknown'}")
    print(f"  InterProScan data dir: {runtime_data or 'unknown'}")

    java_bin = resolve_java_executable(ROOT_DIR)
    print(f"  Java: {java_bin if java_bin else 'not found'}")

    if runtime_python:
        python_path = Path(runtime_python)
        print(f"  Python exists: {'yes' if python_path.exists() else 'no'}")
    if runtime_bin:
        bin_path = Path(runtime_bin)
        print(f"  Bin dir exists: {'yes' if bin_path.exists() else 'no'}")
    if runtime_data:
        data_path = Path(runtime_data)
        print(f"  Data dir exists: {'yes' if data_path.exists() else 'no'}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Install and configure a runnable iTAK3 environment for the current machine.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              ./install_runtime.sh
              ./install_runtime.sh --status
              ./install_runtime.sh --download-db
              ./install_runtime.sh --clean-runtime
              ./install_runtime.sh --with-predict
              ./install_runtime.sh --with-predict --predict-backend mps
              ./install_runtime.sh --with-predict --predict-backend cuda --torch-index-url https://download.pytorch.org/whl/cu121
              ./install_runtime.sh --smoke-test
              ./install_runtime.sh --no-venv --skip-pip
              ./install_runtime.sh --interproscan-dir /path/to/interproscan
              ./install_runtime.sh --venv .venv-itak --torch-index-url https://download.pytorch.org/whl/cpu
            """
        ),
    )
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to bootstrap from")
    parser.add_argument("--venv", default=".venv", help="Virtualenv path to create/use (default: .venv)")
    parser.add_argument("--no-venv", action="store_true", help="Install into the chosen Python directly")
    parser.add_argument("--status", action="store_true", help="Show current local runtime status and exit")
    parser.add_argument("--clean-runtime", action="store_true", help="Remove generated local runtime files and exit")
    parser.add_argument("--remove-venv", action="store_true", help="Also remove the selected venv when used with --clean-runtime")
    parser.add_argument(
        "--download-db",
        action="store_true",
        help="Automatically prepare the bundled db/interproscan when the requested InterProScan is missing or invalid",
    )
    parser.add_argument("--skip-pip", action="store_true", help="Skip pip installation steps")
    parser.add_argument("--skip-check", action="store_true", help="Skip final dependency check")
    parser.add_argument("--with-predict", action="store_true", help="Also install PyTorch prediction dependencies")
    parser.add_argument(
        "--predict-backend",
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Prediction backend to prepare when --with-predict is enabled (default: cpu)",
    )
    parser.add_argument(
        "--torch-index-url",
        default="https://download.pytorch.org/whl/cpu",
        help="PyTorch package index URL when --with-predict is enabled for cpu/cuda installs",
    )
    parser.add_argument(
        "--interproscan-dir",
        default=str(ROOT_DIR / "db" / "interproscan"),
        help="InterProScan installation directory to configure",
    )
    parser.add_argument("--bin-dir", default=None, help="Override helper binary directory")
    parser.add_argument("--data-dir", default=None, help="Override InterProScan data directory")
    parser.add_argument("--java", default=None, help="Explicit Java executable to use")
    parser.add_argument("--smoke-test", action="store_true", help="Run a post-install smoke test")
    parser.add_argument(
        "--smoke-test-input",
        default=str(ROOT_DIR / "test_protein.fasta"),
        help="Input FASTA used for the smoke test",
    )
    parser.add_argument(
        "--smoke-test-appl",
        default="PROSITEPROFILES",
        help="InterProScan application list used during the smoke test",
    )
    parser.add_argument(
        "--smoke-test-output",
        default=str(ROOT_DIR / "output" / "install_smoke"),
        help="Output directory root used during the smoke test",
    )
    return parser


def main():
    args = build_parser().parse_args()

    bootstrap_python = Path(args.python).resolve()
    if not bootstrap_python.exists():
        raise SystemExit(f"Python interpreter not found: {bootstrap_python}")

    requested_interproscan_dir = Path(args.interproscan_dir).resolve()

    venv_dir = (ROOT_DIR / args.venv).resolve()
    if args.status:
        print_status(interproscan_dir=requested_interproscan_dir, venv_dir=venv_dir)
        return

    if args.clean_runtime:
        removed = clean_runtime(
            interproscan_dir=requested_interproscan_dir,
            venv_dir=venv_dir,
            remove_venv=(not args.no_venv and args.remove_venv),
        )
        if removed:
            print("Removed runtime artifacts:")
            for path in removed:
                print(f"  - {path}")
        else:
            print("No generated runtime artifacts were found.")
        return

    interproscan_dir = ensure_usable_interproscan(
        requested_interproscan_dir,
        auto_download=args.download_db,
    )
    config_template = interproscan_dir / "interproscan.properties"
    interproscan_script = interproscan_dir / "interproscan.sh"
    using_bundled_interproscan = interproscan_dir.resolve() == DEFAULT_INTERPROSCAN_DIR.resolve()

    if args.no_venv:
        runtime_python = bootstrap_python
    else:
        runtime_python = create_virtualenv(bootstrap_python, venv_dir)

    helper_root = Path(args.bin_dir).resolve() if args.bin_dir else get_helper_root(ROOT_DIR)
    if helper_root is None:
        helper_root = interproscan_dir / "bin"
    configured_bin_dir = interproscan_dir / "bin"

    data_dir = Path(args.data_dir).resolve() if args.data_dir else (interproscan_dir / "data").resolve()
    if not data_dir.exists():
        raise SystemExit(f"InterProScan data directory not found: {data_dir}")

    java_bin = Path(args.java).resolve() if args.java else resolve_java_executable(ROOT_DIR)
    if java_bin is None:
        raise SystemExit("Java executable not found. Install Java 11+ or place a JDK under .local-jdk/")

    if not args.skip_pip:
        install_python_packages(
            runtime_python,
            with_predict=args.with_predict,
            predict_backend=args.predict_backend,
            torch_index_url=args.torch_index_url,
        )

    if using_bundled_interproscan:
        activated, activation_message = activate_bundled_interproscan_binaries(ROOT_DIR, interproscan_dir)
        print(activation_message)
        if not activated:
            raise SystemExit(activation_message)
    else:
        print(f"Using external InterProScan installation: {interproscan_dir}")

    local_config = interproscan_dir / "interproscan.local.properties"
    write_local_interproscan_config(
        config_template,
        local_config,
        bin_dir=configured_bin_dir,
        data_dir=data_dir,
        python_cmd=runtime_python,
        perl_cmd=detect_perl(),
    )

    wrapper_path = ROOT_DIR / "run_interproscan_local.sh"
    write_interproscan_wrapper(
        wrapper_path,
        java_bin=java_bin,
        helper_root=helper_root,
        config_path=local_config,
        interproscan_script=interproscan_script,
    )
    legacy_wrapper_path = ROOT_DIR / "run_itak2_local.sh"
    if legacy_wrapper_path.exists() or legacy_wrapper_path.is_symlink():
        legacy_wrapper_path.unlink()

    legacy_itak_wrapper_path = ROOT_DIR / "run_itak3_local.sh"
    write_itak_wrapper(
        legacy_itak_wrapper_path,
        java_bin=java_bin,
        helper_root=helper_root,
        config_path=local_config,
        python_bin=runtime_python,
        interproscan_script=(interproscan_script if not using_bundled_interproscan else None),
    )

    print(f"Wrote local InterProScan config: {local_config}")
    print(f"Wrote runtime wrapper: {wrapper_path}")
    print(f"Refreshed deprecated iTAK3 wrapper: {legacy_itak_wrapper_path}")
    print(f"Runtime Python: {runtime_python}")
    print(f"Java: {java_bin}")
    print(f"Configured InterProScan script: {interproscan_script}")
    print(f"InterProScan bin directory: {configured_bin_dir}")
    print(f"Helper binary source: {helper_root}")
    print(f"InterProScan data directory: {data_dir}")
    if args.with_predict:
        print(f"Prediction backend: {args.predict_backend}")
        if args.predict_backend in {'cpu', 'cuda'}:
            print(f"PyTorch index URL: {args.torch_index_url}")

    if not args.skip_check:
        run_dependency_check(
            runtime_python,
            config_path=local_config,
            interproscan_script=(interproscan_script if not using_bundled_interproscan else None),
        )

    if args.smoke_test:
        run_smoke_test(
            runtime_python,
            wrapper_path=wrapper_path,
            config_path=local_config,
            input_fasta=args.smoke_test_input,
            appl_list=args.smoke_test_appl,
            output_root=Path(args.smoke_test_output).resolve(),
            interproscan_script=(interproscan_script if not using_bundled_interproscan else None),
        )

    print("\nInstall completed.")
    print(f"Use iTAK with: {shell_join([str(ROOT_DIR / 'itak'), '-i', 'input.fasta'])}")
    print("Preferred automation target: ./itak")
    if not using_bundled_interproscan:
        print(
            "This runtime uses an external InterProScan. "
            f"Add --interproscan {shlex.quote(str(interproscan_script))} to iTAK commands."
        )
    print(f"Deprecated compatibility wrapper remains available at: {shell_join([str(legacy_itak_wrapper_path), '-i', 'input.fasta'])}")
    print("Do not rely on run_itak3_local.sh for new scripts or documentation.")
    print(f"Use InterProScan directly with: {shell_join([str(wrapper_path), '-i', 'input.fasta', '-f', 'json', '-d', 'output_dir'])}")


if __name__ == "__main__":
    main()
