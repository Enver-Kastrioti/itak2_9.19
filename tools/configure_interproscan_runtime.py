#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configure and validate an InterProScan runtime for iTAK."""

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from module.db_manager import (  # noqa: E402
    validate_interproscan_data_directory,
    validate_interproscan_engine_installation,
)
from module.runtime_tools import (  # noqa: E402
    activate_bundled_interproscan_binaries,
    build_runtime_env,
    get_helper_root,
    resolve_java_executable,
)


DEFAULT_ENGINE_DIR = ROOT_DIR / "db" / "interproscan"
DEFAULT_DATA_DIR = DEFAULT_ENGINE_DIR / "data"
DEFAULT_MANIFEST = ROOT_DIR / "runtime" / "interproscan_manifest.json"
DEFAULT_LOCAL_CONFIG = "interproscan.local.properties"
VERSION_PATTERN = re.compile(r"(\d+\.\d+(?:-\d+\.\d+)?)")


def platform_key():
    current = platform.system().lower()
    if current == "darwin":
        return "macos"
    if current == "linux":
        return "linux"
    return current


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


def update_property_line(line, replacements):
    for key, value in replacements.items():
        prefix = f"{key}="
        if line.startswith(prefix):
            return prefix + value
    return line


def render_local_config(template_path, output_path, *, bin_dir, data_dir, python_cmd, perl_cmd):
    replacements = {
        "bin.directory": str(bin_dir),
        "data.directory": str(data_dir),
        "python3.command": str(python_cmd),
        "perl.command": str(perl_cmd),
    }

    lines = template_path.read_text(encoding="utf-8").splitlines()
    rendered = [update_property_line(line, replacements) for line in lines]
    output_path.write_text("\n".join(rendered) + "\n", encoding="utf-8")


def detect_perl(explicit_perl=None):
    if explicit_perl:
        return Path(explicit_perl).resolve()

    perl_path = shutil.which("perl")
    if perl_path:
        return Path(perl_path).resolve()
    return Path("perl")


def build_interproscan_env(*, java_bin=None, config_path=None):
    env = build_runtime_env(ROOT_DIR)
    if java_bin is not None:
        env["PATH"] = str(java_bin.parent) + os.pathsep + env.get("PATH", "")
    if config_path is not None:
        env["INTERPROSCAN_CONF"] = str(config_path)
    return env


def print_issue_block(title, issues):
    print(f"{title}:")
    if not issues:
        print("  [OK] no issues")
        return
    for issue in issues:
        print(f"  - {issue}")


def validate_engine(engine_dir):
    issues = list(validate_interproscan_engine_installation(engine_dir))
    script_path = engine_dir / "interproscan.sh"
    if script_path.exists() and not os.access(script_path, os.X_OK):
        issues.append(f"InterProScan script is not executable: {script_path}")
    return issues


def validate_data(data_dir):
    return list(validate_interproscan_data_directory(data_dir))


def run_interproscan_version(engine_dir, *, config_path=None, java_bin=None, timeout=60):
    interproscan_script = engine_dir / "interproscan.sh"
    if not interproscan_script.exists():
        return False, None, f"InterProScan script not found: {interproscan_script}"

    env = build_interproscan_env(java_bin=java_bin, config_path=config_path)
    try:
        result = subprocess.run(
            [str(interproscan_script), "-version"],
            capture_output=True,
            text=True,
            cwd=engine_dir,
            env=env,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, None, f"InterProScan self-test timed out: {interproscan_script} -version"
    except Exception as exc:
        return False, None, f"InterProScan self-test could not be executed: {exc}"

    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        if details:
            details = details.splitlines()[0]
        else:
            details = f"exit code {result.returncode}"
        return False, None, f"InterProScan self-test failed: {details}"

    output = (result.stdout or result.stderr or "").strip()
    version_line = output.splitlines()[0] if output else "InterProScan -version completed successfully"
    match = VERSION_PATTERN.search(output)
    version = match.group(1) if match else None
    return True, version, version_line


def check_engine_policy(manifest, *, detected_version, allow_unsupported):
    current_platform = platform_key()
    policy = manifest.get("engine_policy", {}).get(current_platform, {})
    supported_versions = policy.get("supported", [])
    allow_unlisted = bool(policy.get("allow_unlisted", False))

    if not supported_versions:
        return True, None

    if detected_version is None:
        if allow_unlisted or allow_unsupported:
            return True, "Engine version could not be parsed; continuing because unlisted versions are allowed"
        return False, f"Engine version could not be parsed for {current_platform}; supported versions: {', '.join(supported_versions)}"

    if detected_version in supported_versions:
        return True, None

    message = (
        f"Engine version {detected_version} is not in the tested {current_platform} set: "
        f"{', '.join(supported_versions)}"
    )
    if allow_unlisted or allow_unsupported:
        return True, message
    return False, message


def maybe_activate_helpers(manifest, *, engine_dir):
    current_platform = platform_key()
    policy = manifest.get("engine_policy", {}).get(current_platform, {})
    requires_helpers = bool(policy.get("requires_itak_helpers", False))

    if not requires_helpers:
        return True, "No helper activation required for this platform"

    return activate_bundled_interproscan_binaries(ROOT_DIR, engine_dir)


class TemporaryConfigContext:
    def __init__(self, engine_dir, data_dir, *, python_cmd, perl_cmd):
        self.engine_dir = engine_dir
        self.data_dir = data_dir
        self.python_cmd = python_cmd
        self.perl_cmd = perl_cmd
        self.temp_dir = None
        self.config_path = None

    def __enter__(self):
        template_path = self.engine_dir / "interproscan.properties"
        self.temp_dir = tempfile.TemporaryDirectory(prefix="itak-runtime-")
        self.config_path = Path(self.temp_dir.name) / DEFAULT_LOCAL_CONFIG
        render_local_config(
            template_path,
            self.config_path,
            bin_dir=self.engine_dir / "bin",
            data_dir=self.data_dir,
            python_cmd=self.python_cmd,
            perl_cmd=self.perl_cmd,
        )
        return self.config_path

    def __exit__(self, exc_type, exc, exc_tb):
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
        return False


def evaluate_runtime(
    *,
    manifest,
    engine_dir,
    data_dir,
    python_cmd,
    perl_cmd,
    java_bin,
    local_config_path,
    allow_unsupported,
    activate_helpers,
):
    engine_issues = validate_engine(engine_dir)
    data_issues = validate_data(data_dir)
    current_platform = platform_key()
    policy = manifest.get("engine_policy", {}).get(current_platform, {})
    helpers_required = bool(policy.get("requires_itak_helpers", False))

    helper_ok = True
    helper_message = (
        "Helper activation required on this platform but not attempted in read-only mode"
        if helpers_required and not activate_helpers
        else "No helper activation required for this platform"
    )
    if activate_helpers and not engine_issues:
        helper_ok, helper_message = maybe_activate_helpers(manifest, engine_dir=engine_dir)
        if not helper_ok:
            engine_issues.append(helper_message)

    version_ok = False
    detected_version = None
    version_message = "InterProScan self-test skipped"
    policy_ok = False
    policy_message = None

    if not engine_issues and not data_issues:
        with TemporaryConfigContext(
            engine_dir,
            data_dir,
            python_cmd=python_cmd,
            perl_cmd=perl_cmd,
        ) as temp_config:
            version_ok, detected_version, version_message = run_interproscan_version(
                engine_dir,
                config_path=temp_config,
                java_bin=java_bin,
            )
        if version_ok:
            policy_ok, policy_message = check_engine_policy(
                manifest,
                detected_version=detected_version,
                allow_unsupported=allow_unsupported,
            )
        else:
            policy_ok = False

    existing_properties = load_properties(local_config_path)

    return {
        "engine_issues": engine_issues,
        "data_issues": data_issues,
        "helper_ok": helper_ok,
        "helper_message": helper_message,
        "helpers_required": helpers_required,
        "version_ok": version_ok,
        "detected_version": detected_version,
        "version_message": version_message,
        "policy_ok": policy_ok,
        "policy_message": policy_message,
        "existing_properties": existing_properties,
    }


def print_status_report(
    *,
    manifest_path,
    manifest,
    engine_dir,
    data_dir,
    local_config_path,
    helper_root,
    java_bin,
    python_cmd,
    perl_cmd,
    report,
):
    print("InterProScan runtime status:")
    print(f"  Manifest: {manifest_path}")
    print(f"  Platform: {platform_key()}")
    print(f"  Engine dir: {engine_dir}")
    print(f"  Data dir: {data_dir}")
    print(f"  Local config: {local_config_path} ({'present' if local_config_path.exists() else 'missing'})")
    print(f"  Helper root: {helper_root if helper_root else 'not found'}")
    print(f"  Java: {java_bin if java_bin else 'not found'}")
    print(f"  Python command: {python_cmd}")
    print(f"  Perl command: {perl_cmd}")
    print(f"  Manifest db version: {manifest.get('db_version', 'unknown')}")
    print(f"  Manifest helper version: {manifest.get('helper_version', 'unknown')}")
    print(f"  Self-test: {'ok' if report['version_ok'] else 'failed'}")
    if report["detected_version"]:
        print(f"  Detected engine version: {report['detected_version']}")
    print(f"  Self-test detail: {report['version_message']}")
    print(f"  Helper activation: {'required' if report['helpers_required'] else 'not required'}")
    print(f"  Helper detail: {report['helper_message']}")
    if report["policy_message"]:
        print(f"  Engine policy: {report['policy_message']}")
    else:
        print(f"  Engine policy: {'ok' if report['policy_ok'] else 'not satisfied'}")

    properties = report["existing_properties"]
    if properties:
        print("  Existing local config values:")
        for key in ("bin.directory", "data.directory", "python3.command", "perl.command"):
            print(f"    {key}: {properties.get(key, 'missing')}")

    print_issue_block("Engine issues", report["engine_issues"])
    print_issue_block("Data issues", report["data_issues"])


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate and configure the bundled iTAK InterProScan runtime without creating a .venv.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to the iTAK InterProScan runtime manifest",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to record in interproscan.local.properties",
    )
    parser.add_argument(
        "--perl",
        default=None,
        help="Perl interpreter to record in interproscan.local.properties",
    )
    parser.add_argument(
        "--java",
        default=None,
        help="Explicit Java executable to use for the InterProScan self-test",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current runtime status without writing configuration",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate engine, data, and self-test without writing configuration",
    )
    parser.add_argument(
        "--allow-unsupported-engine",
        action="store_true",
        help="Allow an InterProScan engine version outside the manifest tested set",
    )
    return parser


def main():
    deprecated_args = {"--engine-dir", "--data-dir"}
    used_deprecated = sorted(arg for arg in deprecated_args if arg in sys.argv)
    if used_deprecated:
        joined = ", ".join(used_deprecated)
        raise SystemExit(
            f"{joined} is no longer supported. iTAK now always configures the bundled db/interproscan runtime."
        )

    args = build_parser().parse_args()

    engine_dir = DEFAULT_ENGINE_DIR.resolve()
    data_dir = DEFAULT_DATA_DIR.resolve()
    manifest_path = Path(args.manifest).resolve()
    local_config_path = engine_dir / DEFAULT_LOCAL_CONFIG
    helper_root = get_helper_root(ROOT_DIR)
    python_cmd = Path(args.python).resolve()
    perl_cmd = detect_perl(args.perl)
    java_bin = Path(args.java).resolve() if args.java else resolve_java_executable(ROOT_DIR)

    if not manifest_path.exists():
        raise SystemExit(f"Runtime manifest not found: {manifest_path}")
    if not python_cmd.exists():
        raise SystemExit(f"Python interpreter not found: {python_cmd}")
    if args.java and not java_bin.exists():
        raise SystemExit(f"Java executable not found: {java_bin}")
    if java_bin is None:
        raise SystemExit("Java executable not found. Install Java 11+ or pass --java.")

    manifest = load_manifest(manifest_path)
    report = evaluate_runtime(
        manifest=manifest,
        engine_dir=engine_dir,
        data_dir=data_dir,
        python_cmd=python_cmd,
        perl_cmd=perl_cmd,
        java_bin=java_bin,
        local_config_path=local_config_path,
        allow_unsupported=args.allow_unsupported_engine,
        activate_helpers=(not args.status and not args.check),
    )

    if args.status:
        print_status_report(
            manifest_path=manifest_path,
            manifest=manifest,
            engine_dir=engine_dir,
            data_dir=data_dir,
            local_config_path=local_config_path,
            helper_root=helper_root,
            java_bin=java_bin,
            python_cmd=python_cmd,
            perl_cmd=perl_cmd,
            report=report,
        )
        return

    if args.check:
        print_status_report(
            manifest_path=manifest_path,
            manifest=manifest,
            engine_dir=engine_dir,
            data_dir=data_dir,
            local_config_path=local_config_path,
            helper_root=helper_root,
            java_bin=java_bin,
            python_cmd=python_cmd,
            perl_cmd=perl_cmd,
            report=report,
        )
        if report["engine_issues"] or report["data_issues"] or not report["version_ok"] or not report["policy_ok"]:
            raise SystemExit(1)
        return

    if report["engine_issues"]:
        print_issue_block("Engine issues", report["engine_issues"])
        raise SystemExit(1)
    if report["data_issues"]:
        print_issue_block("Data issues", report["data_issues"])
        raise SystemExit(1)
    if not report["helper_ok"]:
        raise SystemExit(report["helper_message"])
    if not report["version_ok"]:
        raise SystemExit(report["version_message"])
    if not report["policy_ok"]:
        raise SystemExit(report["policy_message"] or "Engine policy validation failed")

    template_path = engine_dir / "interproscan.properties"
    render_local_config(
        template_path,
        local_config_path,
        bin_dir=engine_dir / "bin",
        data_dir=data_dir,
        python_cmd=python_cmd,
        perl_cmd=perl_cmd,
    )

    version_ok, detected_version, version_message = run_interproscan_version(
        engine_dir,
        config_path=local_config_path,
        java_bin=java_bin,
    )
    if not version_ok:
        raise SystemExit(version_message)

    print("Configured InterProScan runtime:")
    print(f"  Engine dir: {engine_dir}")
    print(f"  Data dir: {data_dir}")
    print(f"  Local config: {local_config_path}")
    print(f"  Python command: {python_cmd}")
    print(f"  Perl command: {perl_cmd}")
    print(f"  Java: {java_bin}")
    if report["helper_message"]:
        print(f"  Helper detail: {report['helper_message']}")
    print(f"  Engine version: {detected_version or 'unknown'}")
    print(f"  Self-test detail: {version_message}")


if __name__ == "__main__":
    main()
