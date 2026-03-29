#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import subprocess
from pathlib import Path
from Bio import SeqIO
import csv
from datetime import datetime
import time
import json
import shutil
import tarfile
from dataclasses import dataclass

# Import dependency checker module
try:
    from module.check_dependencies import DependencyChecker
except ImportError:
    print("Warning: Unable to import the dependency checker module; dependency checks will be skipped")
    DependencyChecker = None

try:
    from module.runtime_tools import (
        activate_bundled_interproscan_binaries,
        build_runtime_env,
        resolve_helper_executable,
        resolve_java_executable,
    )
except ImportError:
    print("Warning: Unable to import runtime tool helpers; platform-specific binary setup will be limited")
    activate_bundled_interproscan_binaries = None
    build_runtime_env = None
    resolve_helper_executable = None
    resolve_java_executable = None

try:
    from module.output_contracts import (
        build_tftr_match_record,
        load_pk_records_from_tsv,
        load_tftr_records_from_table,
        records_to_classification_result,
        write_combined_summary,
        write_tftr_outputs,
    )
except ImportError:
    print("Warning: Unable to import output contract helpers; normalized output handling will be limited")
    build_tftr_match_record = None
    load_pk_records_from_tsv = None
    load_tftr_records_from_table = None
    records_to_classification_result = None
    write_combined_summary = None
    write_tftr_outputs = None

# Import FASTA validation module
try:
    from module.validate_fasta import FastaValidator
except ImportError:
    print("Warning: Unable to import the FASTA validation module; FASTA validation will be skipped")
    FastaValidator = None

# Script directory
SCRIPT_DIR = Path(__file__).parent.absolute()
PREDICT_SCRIPT = SCRIPT_DIR / "pre_model" / "predict.py"
MODULE_DIR = SCRIPT_DIR / "module"

# DB paths and automated extraction
DB_DIR = SCRIPT_DIR / "db"
DB_ARCHIVE = SCRIPT_DIR / "db.tar.gz"
INTERPROSCAN_SCRIPT = DB_DIR / "interproscan" / "interproscan.sh"


@dataclass(frozen=True)
class PipelineContext:
    input_fasta: Path
    project_output: Path
    processed_fasta: Path
    preclass_dir: Path
    prediction_csv: Path
    prediction_tf_only_csv: Path
    tf_fasta: Path
    interproscan_dir: Path
    hmmscan_dir: Path
    protein_kinase_dir: Path
    result_dir: Path
    getrule_json: Path

    def resolve_tf_fasta(self):
        if self.tf_fasta.exists():
            return self.tf_fasta

        candidates = sorted(self.preclass_dir.glob("*_tf_sequences.fasta"))
        if len(candidates) == 1:
            return candidates[0]
        return None

    def resolve_analysis_fasta(self, use_predicted):
        if use_predicted:
            return self.resolve_tf_fasta()
        return self.processed_fasta

    def resolve_interpro_json(self, analysis_fasta):
        if analysis_fasta is not None:
            candidate = self.interproscan_dir / f"{Path(analysis_fasta).name}.json"
            if candidate.exists():
                return candidate

        original_json = self.interproscan_dir / f"{self.input_fasta.name}.json"
        if original_json.exists():
            return original_json

        json_files = list(self.interproscan_dir.glob("*.json"))
        if len(json_files) == 1:
            return json_files[0]

        if analysis_fasta is not None:
            return self.interproscan_dir / f"{Path(analysis_fasta).name}.json"
        return original_json

def ensure_db_extracted():
    """
    Ensure the db directory is available.

    If the db directory does not exist, attempt extraction from an archive or download via GitHub.
    This is implemented via module/db_manager.py.
    
    Returns:
        bool: True if the db directory is available; otherwise False.
    """
    try:
        import importlib.util
        db_manager_path = MODULE_DIR / "db_manager.py"
        
        if not db_manager_path.exists():
            print(f"Error: db_manager module not found: {db_manager_path}")
            # Fallback to legacy behavior (existence check only)
            if DB_DIR.exists() and DB_DIR.is_dir():
                return True
            return False
            
        spec = importlib.util.spec_from_file_location("db_manager", db_manager_path)
        db_manager = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(db_manager)
        
        return db_manager.setup_db(SCRIPT_DIR)
            
    except Exception as e:
        print(f"Error while checking/preparing the db directory: {e}")
        return False


# Basic utility functions
def call_module_function(module_name, function_name, *args, **kwargs):

    try:
        # Build module path
        module_path = MODULE_DIR / f"{module_name}.py"
        
        if not module_path.exists():
            raise FileNotFoundError(f"Module file not found: {module_path}")
        
        # Dynamically import module
        import importlib.util
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Resolve function
        if not hasattr(module, function_name):
            raise AttributeError(f"Function {function_name} not found in module {module_name}")
        
        func = getattr(module, function_name)
        
        # Call function
        return func(*args, **kwargs)
        
    except Exception as e:
        print(f"Error calling module function: {e}")
        return None
# Timer formatting
def format_duration(seconds):

    if seconds < 60:
        return f"{seconds:.2f} s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes} min {remaining_seconds:.2f} s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours} h {remaining_minutes} min {remaining_seconds:.2f} s"

def setup_project_output(fasta_file, output=None):

    if output:
        project_output = Path(output)
    else:
        # Use default output directory structure
        fasta_path = Path(fasta_file)
        base_name = fasta_path.stem
        output_base = SCRIPT_DIR / "output"
        
        # Create a unique output directory
        counter = 0
        while True:
            if counter == 0:
                project_output = output_base / base_name
            else:
                project_output = output_base / f"{base_name}_{counter}"
            
            if not project_output.exists():
                break
            counter += 1
    
    # Create output directory
    project_output.mkdir(parents=True, exist_ok=True)
    return project_output


def _load_module_from_path(module_name, module_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_get_fasta_module():
    return _load_module_from_path("get_fasta", MODULE_DIR / "get_fasta.py")


def _resolve_processed_fasta_path(fasta_file, project_output):
    try:
        module = _load_get_fasta_module()
        return Path(module.get_processed_fasta_path(str(fasta_file), project_output))
    except Exception:
        return Path(project_output) / f"{Path(fasta_file).stem}_protein_replaced.fasta"


def _ensure_processed_fasta(context):
    processed_fasta = context.processed_fasta
    if processed_fasta.exists():
        return processed_fasta

    try:
        module = _load_get_fasta_module()
        output_path = module.generate_protein_fasta_with_translation(
            str(context.input_fasta),
            output_dir=context.project_output,
        )
        if output_path:
            return Path(output_path)
    except Exception:
        pass

    return context.input_fasta


def build_pipeline_context(fasta_file, output=None, project_output=None):
    input_fasta = Path(fasta_file)
    project_output = Path(project_output) if project_output else setup_project_output(str(input_fasta), output)
    processed_fasta = _resolve_processed_fasta_path(input_fasta, project_output)
    processed_stem = processed_fasta.stem

    preclass_dir = project_output / "protein_model_preclassification"
    interproscan_dir = project_output / "InterproScan"
    hmmscan_dir = project_output / "hmmscan"
    protein_kinase_dir = project_output / "protein_kinase"
    result_dir = project_output / "result"

    return PipelineContext(
        input_fasta=input_fasta,
        project_output=project_output,
        processed_fasta=processed_fasta,
        preclass_dir=preclass_dir,
        prediction_csv=preclass_dir / f"{input_fasta.stem}_prediction.csv",
        prediction_tf_only_csv=preclass_dir / f"{input_fasta.stem}_prediction_tf_only.csv",
        tf_fasta=preclass_dir / f"{processed_stem}_tf_sequences.fasta",
        interproscan_dir=interproscan_dir,
        hmmscan_dir=hmmscan_dir,
        protein_kinase_dir=protein_kinase_dir,
        result_dir=result_dir,
        getrule_json=project_output / "getrule.json",
    )


def _format_step_heading(step_number, label):
    if step_number is None:
        return f"\n{label}"
    return f"\n{step_number}. {label}"


def _build_prediction_command(
    fasta_file,
    threshold,
    output_csv,
    project_output,
    predict_mode="fast",
    debug=False,
    use_supplementary=False,
    supplementary_only=False,
    supp_models=None,
    grad_cam_mode="none",
):
    if not PREDICT_SCRIPT.exists():
        raise FileNotFoundError(f"Prediction script not found: {PREDICT_SCRIPT}")

    cmd = [
        sys.executable,
        str(PREDICT_SCRIPT),
        "--fasta",
        str(Path(fasta_file).absolute()),
        "--threshold",
        str(threshold),
        "--output",
        str(Path(output_csv).absolute()),
        "--project-output",
        str(Path(project_output).absolute()),
        "--mode",
        predict_mode,
    ]

    if supplementary_only:
        cmd.append("--supplementary-only")
    elif use_supplementary:
        cmd.append("--use-supplementary")

    if supp_models:
        cmd.append("--supp-models")
        for model_name in supp_models:
            cmd.append(model_name)

    if grad_cam_mode != "none":
        cmd.extend(["--grad-cam-mode", grad_cam_mode])

    if debug:
        cmd.append("--debug")

    return cmd


def _write_combined_result_summary(project_output, tf_records=None, pk_records=None):
    project_output = Path(project_output)
    result_dir = project_output / "result"
    result_dir.mkdir(exist_ok=True)

    summary_path = result_dir / "all_match_tbl.txt"
    if tf_records is None:
        if load_tftr_records_from_table is not None:
            tf_records = load_tftr_records_from_table(result_dir / "match_tbl.txt")
        else:
            tf_records = {}
    if pk_records is None:
        if load_pk_records_from_tsv is not None:
            pk_records = load_pk_records_from_tsv(project_output / "protein_kinase" / "pk_classification.tsv")
        else:
            pk_records = {}

    if write_combined_summary is not None:
        write_combined_summary(tf_records, pk_records, summary_path)
    else:
        summary_path.write_text("", encoding="utf-8")

    print(
        "Combined result summary written: "
        f"{summary_path} ({len(set(tf_records.keys()) | set(pk_records.keys()))} sequences; "
        f"{len(tf_records)} TF/TR, {len(pk_records)} PK)"
    )
    return True

def cleanup_processed_fasta(project_output, fasta_file):
    processed_fasta = _resolve_processed_fasta_path(fasta_file, project_output)

    try:
        if processed_fasta.exists() and processed_fasta.is_file():
            processed_fasta.unlink()
            return True
    except Exception:
        return False
    return False

def resolve_existing_project_output(fasta_file, output=None):
    if output:
        return Path(output)

    output_base = SCRIPT_DIR / "output"
    stem = Path(fasta_file).stem
    candidates = [p for p in output_base.glob(f"{stem}*") if p.is_dir()]
    if not candidates:
        return output_base / stem
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]

# Analysis-module pipeline
def run_analysis_modules(context, use_predicted=True, debug=False, score_threshold=1.0, classification_mode='specific'):
    """
    Run analysis modules for transcription factor family classification.
    
    Args:
        context (PipelineContext): Pipeline context with resolved paths.
        use_predicted (bool): Whether to use predicted TF sequences.
        debug (bool): Whether to enable debug mode and write intermediate artifacts.
        score_threshold (float): Score threshold for filtering InterProScan results.
        classification_mode (str): Classification mode ('specific' or 'score').
    
    Returns:
        bool: True if the analysis completes successfully; otherwise False.
    """
    print(f"\n{'='*50}")
    print("Starting analysis modules...")
    print(f"{'='*50}")
    
    try:
        analysis_path = context.resolve_analysis_fasta(use_predicted)
        if use_predicted:
            if analysis_path is None:
                print(f"Warning: predicted TF FASTA file not found: {context.tf_fasta}")
                print("Falling back to the original input for analysis")
                analysis_path = context.input_fasta
            elif analysis_path != context.tf_fasta:
                print(f"Warning: predicted TF FASTA file not found: {context.tf_fasta}")
                print(f"Using detected TF FASTA instead: {analysis_path}")
        else:
            analysis_path = _ensure_processed_fasta(context)

        ipr_file = context.resolve_interpro_json(analysis_path)
        hmmscan_file = context.hmmscan_dir / "result.tbl"
        rule_file = SCRIPT_DIR / "rule.txt"
        
        # Check required files
        missing_files = []
        if not ipr_file.exists():
            missing_files.append(str(ipr_file))
        if not hmmscan_file.exists():
            missing_files.append(str(hmmscan_file))
        if not rule_file.exists():
            missing_files.append(str(rule_file))
        
        if missing_files:
            print("Error: the following required files are missing:")
            for file in missing_files:
                print(f"  - {file}")
            return False
        
        print("Inputs:")
        print(f"  FASTA: {analysis_path}")
        print(f"  IPR JSON: {ipr_file}")
        print(f"  hmmscan: {hmmscan_file}")
        print(f"  Rule file: {rule_file}")
        
        # Create result output directory (under the project directory)
        result_dir = context.result_dir
        result_dir.mkdir(exist_ok=True)
        print(f"Results will be written to: {result_dir}")
        
        # Step 0: run get_rule and write rule JSON under the project directory
        print("\nStep 0: running get_rule...")
        getrule_success = run_getrule_module(str(rule_file), str(context.project_output), debug=debug)
        if not getrule_success:
            print("get_rule failed")
            return False
        
        # Step 1: run get_json (supports in-memory data passing)
        print("\nStep 1: running get_json...")
        
        jsonbuild_result = run_jsonbuild_module(str(ipr_file), result_dir, debug=debug, score_threshold=score_threshold)
        
        # Evaluate get_json results
        if isinstance(jsonbuild_result, tuple) and len(jsonbuild_result) == 3:
            jsonbuild_success, filtered_data, raw_data = jsonbuild_result
            if not jsonbuild_success:
                print("get_json failed")
                return False
            print("get_json returned in-memory results successfully")
        else:
            # Backward-compatible return format
            jsonbuild_success = jsonbuild_result
            if not jsonbuild_success:
                print("get_json failed")
                return False
            filtered_data = None
            raw_data = None
            print("get_json is using file-based data passing")
        
        # Step 2: run selfbuild_hmm
        print("\nStep 2: running selfbuild_hmm...")
        specpfam_result = run_specpfam_module(str(hmmscan_file), result_dir, debug=debug)
        
        # Evaluate selfbuild_hmm results
        if isinstance(specpfam_result, tuple):
            specpfam_success, spec_data = specpfam_result
        else:
            # Backward-compatible return format
            specpfam_success = specpfam_result
            spec_data = None
            
        if not specpfam_success:
            print("selfbuild_hmm failed")
            return False
            
        if spec_data is not None:
            print("Obtained selfbuild_hmm results for classification")
        else:
            print("No selfbuild_hmm results available; classification will rely on IPR data only")
        
        # Step 3: run classification (supports in-memory data passing)
        print("\nStep 3: running classification...")
        class_tf_result = run_class_tf_module(
            str(analysis_path),
            str(rule_file), 
            result_dir, 
            debug=debug,
            filtered_data=filtered_data,
            spec_data=spec_data,
            classification_mode=classification_mode
        )
        if isinstance(class_tf_result, tuple):
            class_tf_success, tf_records = class_tf_result
        else:
            class_tf_success = class_tf_result
            tf_records = {}

        if not class_tf_success:
            print("classification failed")
            return False

        _write_combined_result_summary(context.project_output, tf_records=tf_records)
        
        print(f"\n{'='*50}")
        print("All analysis modules completed")
        print(f"Results saved to: {result_dir}")
        print(f"{'='*50}")
        
        return True
        
    except Exception as e:
        print(f"Error while running analysis modules: {e}")
        return False

def run_getrule_module(rule_file, output_dir, debug=False):
    try:
        import importlib.util
        get_rule_path = MODULE_DIR / "get_rule.py"
        spec = importlib.util.spec_from_file_location("get_rule", get_rule_path)
        get_rule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(get_rule)
        rules = get_rule.parse_rule_file(rule_file)
        if debug:
            output_file = os.path.join(output_dir, 'getrule.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(rules, f, indent=2, ensure_ascii=False)
            print(f"Rules were parsed successfully and saved to {output_file}")
        print("get_rule completed successfully")
        return True
    except Exception as e:
        print(f"Error while running get_rule: {e}")
        return False
# Extract information from the InterProScan JSON results
def run_jsonbuild_module(ipr_file, result_dir, debug=False, score_threshold=1.0):

    try:
        # First try direct function call (in-memory data passing)
        try:
            # Dynamically import get_json
            import importlib.util
            get_json_path = MODULE_DIR / "get_json.py"
            spec = importlib.util.spec_from_file_location("get_json", get_json_path)
            get_json = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(get_json)
            
            print("Running get_json with in-memory data passing...")
            
            # Call process_data
            filtered_result, new_result = get_json.process_data(
                input_file=str(ipr_file),
                output_dir=str(result_dir),  # Always write files because classification depends on them
                debug=debug,
                score_threshold=score_threshold
            )
            
            if filtered_result is not None:
                print("get_json completed successfully (in-memory)")
                return True, filtered_result, new_result
            else:
                print("get_json processing failed")
                return False, None, None
                
        except Exception as e:
            print(f"In-memory path failed; falling back to CLI execution: {e}")
            
            # Fallback to CLI execution
            get_json_script = MODULE_DIR / "get_json.py"
            
            # Build CLI arguments
            cmd = [
                sys.executable,
                str(get_json_script),
                "-i", str(ipr_file),
                "-o", str(result_dir),
                "--score", str(score_threshold)
            ]
            
            # Add --debug if requested
            if debug:
                cmd.append("--debug")
            
            print(f"Running command: {' '.join(cmd)}")
            
            # Execute get_json
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
            
            if result.returncode == 0:
                print("get_json completed successfully (CLI)")
                if result.stdout:
                    print(result.stdout.strip())
                return True, None, None  # CLI mode does not return in-memory results
            else:
                print("get_json failed")
                if result.stderr:
                    print(f"Error output: {result.stderr}")
                if result.stdout:
                    print(f"Standard output: {result.stdout}")
                return False, None, None
            
    except Exception as e:
        print(f"Error while running get_json: {e}")
        return False, None, None

def run_specpfam_module(hmmscan_file, result_dir, debug=False):
    """Run selfbuild_hmm."""
    try:
        # Import and call selfbuild_hmm
        import importlib.util
        selfbuild_hmm_path = MODULE_DIR / "selfbuild_hmm.py"
        spec = importlib.util.spec_from_file_location("selfbuild_hmm", selfbuild_hmm_path)
        selfbuild_hmm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selfbuild_hmm)
        
        # Parse hmmscan results
        result = selfbuild_hmm.parse_pfam_spec(hmmscan_file)
        
        # Write pfamspec.json only in debug mode
        if debug:
            output_file = os.path.join(result_dir, 'pfamspec.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"selfbuild_hmm completed; results saved to: {output_file}")
        else:
            print("selfbuild_hmm completed (debug disabled; pfamspec.json not written)")
        
        print("selfbuild_hmm completed successfully")
        return True, result  # Return status and parsed results
            
    except Exception as e:
        print(f"Error while running selfbuild_hmm: {e}")
        return False

# Run classification
def run_class_tf_module(fasta_file, rule_file, result_dir, debug=False, filtered_data=None, spec_data=None, classification_mode='specific'):

    try:
        # First try direct function call (in-memory data passing)
        try:
            # Dynamically import classification
            import importlib.util
            classification_path = MODULE_DIR / "classification.py"
            spec = importlib.util.spec_from_file_location("classification", classification_path)
            classification = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(classification)
            
            print("Running classification with in-memory data passing...")
            
            # Call process_with_data
            classification_result = classification.process_with_data(
                result_directory=str(result_dir),
                rule_file=str(rule_file),
                filtered_data=filtered_data,
                spec_data=spec_data,
                debug=debug,
                mode=classification_mode
            )
            
            if classification_result is not None:
                normalized_records = {}
                if build_tftr_match_record is not None:
                    for gene_id, data in classification_result.items():
                        normalized_records[gene_id] = build_tftr_match_record(gene_id, data)
                else:
                    normalized_records = classification_result

                if write_tftr_outputs is not None and records_to_classification_result is not None:
                    written_outputs = write_tftr_outputs(normalized_records, result_dir, debug=debug)
                    tbl_path = written_outputs["table"]
                    json_path = written_outputs.get("json")
                else:
                    tbl_path = os.path.join(result_dir, 'match_tbl.txt')
                    with open(tbl_path, 'w', encoding='utf-8') as f:
                        for gene_id, data in classification_result.items():
                            desc_str = ';'.join(data['desc']) if data['desc'] else 'NA'
                            line = f"{gene_id}\t{data['name']}\t{data['family']}\t{data['type']}\t{desc_str}\t{data['other_family']}\n"
                            f.write(line)
                    json_path = os.path.join(result_dir, 'match.json') if debug else None
                    if debug:
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(classification_result, f, indent=2, ensure_ascii=False)

                # Generate classified FASTA
                try:
                    get_fasta_path = MODULE_DIR / "get_fasta.py"
                    spec = importlib.util.spec_from_file_location("get_fasta", get_fasta_path)
                    get_fasta = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(get_fasta)

                    classified_result = (
                        records_to_classification_result(normalized_records)
                        if records_to_classification_result is not None and build_tftr_match_record is not None
                        else classification_result
                    )

                    if classified_result:
                        classified_fasta_path = get_fasta.generate_classified_fasta(
                            fasta_file,
                            classified_result,
                            output_dir=result_dir
                        )
                        if classified_fasta_path:
                            print(f"Classified FASTA generated: {classified_fasta_path}")
                        else:
                            print("Failed to generate classified FASTA")
                    else:
                        print("No classification results; skipping classified FASTA generation")
                except Exception as e:
                    print(f"Error while generating classified FASTA: {e}")
                
                if debug:
                    print("classification completed (in-memory)")
                    print("Results saved to:")
                    print(f"  JSON: {json_path}")
                    print(f"  Table: {tbl_path}")
                else:
                    print("classification completed (in-memory)")
                    print("Results saved to:")
                    print(f"  Table: {tbl_path}")
                
                return True, normalized_records if build_tftr_match_record is not None else classification_result
            else:
                print("classification processing failed")
                return False, {}
                
        except Exception as e:
            print(f"Error while running classification: {e}")
            return False, {}
            
    except Exception as e:
        print(f"Error while running classification: {e}")
        return False, {}

# ============================================================================
# TF sequence extraction
# ============================================================================

def extract_tf_sequences_from_memory(fasta_file, tf_headers, output_dir):
    """
    Extract TF sequences from an in-memory header list.
    
    Args:
        fasta_file (str): Input FASTA file path.
        tf_headers (list): List of TF sequence headers.
        output_dir (str): Output directory.
    
    Returns:
        str: Output FASTA path; None on failure.
    """
    try:
        print("=== Extract TF sequences (in-memory) ===")
        print(f"TFs passing threshold: {len(tf_headers)}")
        
        if not tf_headers:
            print("No TF sequences to extract")
            return None
        
        # Create output file path
        fasta_basename = Path(fasta_file).stem
        output_fasta = Path(output_dir) / f"{fasta_basename}_tf_sequences.fasta"
        
        # Extract sequences
        extracted_count = 0
        with open(output_fasta, 'w') as output_handle:
            with open(fasta_file, 'r') as input_handle:
                for record in SeqIO.parse(input_handle, "fasta"):
                    # Match by record ID or description
                    if record.id in tf_headers or record.description in tf_headers:
                        SeqIO.write(record, output_handle, "fasta")
                        extracted_count += 1
        
        print(f"Extracted {extracted_count} TF sequences successfully")
        
        if extracted_count > 0:
            print(f"TF FASTA saved to: {output_fasta}")
            return str(output_fasta)
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting TF sequences: {e}")
        return None
# Extract sequences
def extract_tf_sequences_from_csv(fasta_file, csv_file, output_dir, threshold):
    try:
        # Read TF headers from CSV
        tf_headers = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['Predicted_Class'] == 'TF' and float(row['TF_Probability']) >= threshold:
                    tf_headers.append(row['Header'])
        
        print("=== Extract TF sequences ===")
        print(f"TFs passing threshold: {len(tf_headers)}")
        
        if not tf_headers:
            print("No TF sequences to extract")
            return None
        
        # Create output file path
        fasta_basename = Path(fasta_file).stem
        output_fasta = Path(output_dir) / f"{fasta_basename}_tf_sequences.fasta"
        
        # Extract sequences
        extracted_count = 0
        with open(output_fasta, 'w') as output_handle:
            with open(fasta_file, 'r') as input_handle:
                for record in SeqIO.parse(input_handle, "fasta"):
                    # Match by record ID or description
                    if record.id in tf_headers or record.description in tf_headers:
                        SeqIO.write(record, output_handle, "fasta")
                        extracted_count += 1
        
        print(f"Extracted {extracted_count} TF sequences successfully")
        
        if extracted_count > 0:
            print(f"TF FASTA saved to: {output_fasta}")
            return str(output_fasta)
        else:
            return None
            
    except Exception as e:
        print(f"Error extracting TF sequences: {e}")
        return None

def _get_or_create_processed_fasta(fasta_file, output_dir):
    context = build_pipeline_context(fasta_file, project_output=output_dir)
    return str(_ensure_processed_fasta(context))


def _run_protein_kinase_analysis(fasta_file, project_output, enabled=True, debug=False):
    if not enabled:
        print("Protein kinase analysis skipped (--skip-pk)")
        return True

    result = call_module_function(
        "protein_kinase",
        "run_protein_kinase_pipeline",
        str(fasta_file),
        str(project_output),
        debug=debug,
    )
    if not isinstance(result, dict) or not result.get("success"):
        print("Protein kinase analysis failed")
        return False

    print(
        "Protein kinase analysis completed: "
        f"{result.get('count', 0)} sequences classified"
    )
    print(f"Protein kinase output: {result.get('output_dir', Path(project_output) / 'protein_kinase')}")
    return True

def _load_prediction_rows(prediction_csv):
    rows = []
    with open(prediction_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            if "Header" not in row or "TF_Probability" not in row:
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"Prediction CSV is empty or malformed: {prediction_csv}")
    return rows

def _write_threshold_prediction_files(rows, out_prediction_csv, out_tf_only_csv, threshold):
    out_prediction_csv = Path(out_prediction_csv)
    out_tf_only_csv = Path(out_tf_only_csv)
    out_prediction_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_prediction_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Header", "Predicted_Class", "TF_Probability", "Non_TF_Probability", "Confidence"])
        for r in rows:
            tf_prob = float(r["TF_Probability"])
            non_tf_prob = float(r.get("Non_TF_Probability", 1.0 - tf_prob))
            conf = float(r.get("Confidence", max(tf_prob, non_tf_prob)))
            pred_class = "TF" if tf_prob >= threshold else "Non-TF"
            writer.writerow([r["Header"], pred_class, f"{tf_prob:.4f}", f"{non_tf_prob:.4f}", f"{conf:.4f}"])

    with open(out_tf_only_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Header", "TF_Probability", "Confidence"])
        for r in rows:
            tf_prob = float(r["TF_Probability"])
            if tf_prob >= threshold:
                non_tf_prob = float(r.get("Non_TF_Probability", 1.0 - tf_prob))
                conf = float(r.get("Confidence", max(tf_prob, non_tf_prob)))
                writer.writerow([r["Header"], f"{tf_prob:.4f}", f"{conf:.4f}"])

def _extract_sequences_by_headers(source_fasta, headers, output_fasta):
    headers_set = set(headers)
    output_fasta = Path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    extracted_count = 0
    with open(output_fasta, "w") as out_handle:
        with open(source_fasta, "r") as in_handle:
            for record in SeqIO.parse(in_handle, "fasta"):
                if record.id in headers_set or record.description in headers_set:
                    SeqIO.write(record, out_handle, "fasta")
                    extracted_count += 1
    return extracted_count

def _unique_output_dir(path):
    path = Path(path)
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.parent / f"{path.name}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1

def _run_prediction_once(fasta_file, threshold, output_csv, project_output, predict_mode="fast", debug=False, use_supplementary=False, supplementary_only=False, supp_models=None):
    cmd = _build_prediction_command(
        fasta_file=fasta_file,
        threshold=threshold,
        output_csv=output_csv,
        project_output=project_output,
        predict_mode=predict_mode,
        debug=debug,
        use_supplementary=use_supplementary,
        supplementary_only=supplementary_only,
        supp_models=supp_models,
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
        env=build_runtime_env(SCRIPT_DIR) if build_runtime_env else None,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Prediction failed")
    if result.stdout:
        print(result.stdout)

def list_predict_transcription_factors(fasta_file, output=None, appl_list=None, debug=False, score_threshold=1.0, classification_mode="score", predict_mode="fast", interproscan_path=None, use_supplementary=False, supplementary_only=False, supp_models=None, run_protein_kinase_analysis=False):
    thresholds = [None, 0.1, 0.3, 0.5, 0.7, 0.9]

    fasta_basename = Path(fasta_file).stem
    output_base = Path(output) if output else (SCRIPT_DIR / "output")
    output_base.mkdir(parents=True, exist_ok=True)

    outputs = {}
    outputs[None] = _unique_output_dir(output_base / f"{fasta_basename}_no_pre")
    for t in thresholds:
        if t is None:
            continue
        outputs[t] = _unique_output_dir(output_base / f"{fasta_basename}_{int(t*100)}")

    cache_output = _unique_output_dir(output_base / f"_{fasta_basename}_list_predict_cache")
    cache_output.mkdir(parents=True, exist_ok=True)
    cache_preclass = cache_output / "protein_model_preclassification"
    cache_preclass.mkdir(exist_ok=True)

    base_prediction_csv = cache_preclass / f"{fasta_basename}_prediction.csv"

    print("\n=== List Predict Mode ===")
    print(f"Input file: {fasta_file}")
    print(f"Output root: {output_base}")
    print(f"Thresholds: {['NO_PREDICT' if t is None else int(t*100) for t in thresholds]}")
    if run_protein_kinase_analysis:
        print("Note: protein kinase analysis is currently skipped in --list-predict mode")

    _run_prediction_once(
        fasta_file=fasta_file,
        threshold=0.1,
        output_csv=base_prediction_csv,
        project_output=cache_output,
        predict_mode=predict_mode,
        debug=debug,
        use_supplementary=use_supplementary,
        supplementary_only=supplementary_only,
        supp_models=supp_models,
    )

    cache_context = build_pipeline_context(fasta_file, project_output=cache_output)
    processed_fasta = str(preprocess_input_sequences(cache_context))
    rows = _load_prediction_rows(base_prediction_csv)

    for t in thresholds:
        if t is None:
            child_output = outputs[None]
            print(f"\n=== Output: {child_output.name} ===")
            success = analyze_sequences_directly(
                fasta_file=fasta_file,
                output=str(child_output),
                appl_list=appl_list,
                debug=debug,
                score_threshold=score_threshold,
                classification_mode=classification_mode,
                interproscan_path=interproscan_path,
                run_protein_kinase_analysis=False,
            )
            if not success:
                return False
            continue

        child_output = outputs[t]
        print(f"\n=== Output: {child_output.name} (threshold={t}) ===")
        context = build_pipeline_context(fasta_file, project_output=child_output)
        project_output = context.project_output
        preclass_dir = context.preclass_dir
        preclass_dir.mkdir(exist_ok=True)

        prediction_csv = context.prediction_csv
        tf_only_csv = context.prediction_tf_only_csv
        _write_threshold_prediction_files(rows, prediction_csv, tf_only_csv, t)

        tf_headers = [r["Header"] for r in rows if float(r["TF_Probability"]) >= t]
        tf_fasta_path = preclass_dir / f"{fasta_basename}_tf_sequences.fasta"
        extracted = _extract_sequences_by_headers(processed_fasta, tf_headers, tf_fasta_path)
        print(f"Extracted TF sequences: {extracted} / {len(tf_headers)}")

        if extracted == 0:
            (project_output / "result").mkdir(exist_ok=True)
            tbl_path = project_output / "result" / "match_tbl.txt"
            tbl_path.write_text("", encoding="utf-8")
            continue

        analysis_success = run_tf_tr_analysis_pipeline(
            context,
            analysis_fasta=tf_fasta_path,
            use_predicted=True,
            appl_list=appl_list,
            debug=debug,
            score_threshold=score_threshold,
            classification_mode=classification_mode,
            interproscan_path=interproscan_path,
            run_interproscan_analysis=True,
            run_hmmscan_analysis=True,
            step_numbers=(3, 4, 5),
        )
        if not analysis_success:
            return False

    return True

# InterProScan functionality

def run_interproscan(fasta_file, output_dir, appl_list=None, interproscan_path=None):
    try:
        # Ensure db directory is available
        if not interproscan_path and not ensure_db_extracted():
            print("Error: db directory is not available")
            return False
            
        # Create InterProScan output directory
        ipr_output_dir = Path(output_dir) / "InterproScan"
        ipr_output_dir.mkdir(exist_ok=True)
        
        # Build InterProScan command
        if interproscan_path:
             interproscan_script = str(interproscan_path)
        else:
             interproscan_script = str(INTERPROSCAN_SCRIPT)

        if (
            activate_bundled_interproscan_binaries is not None
            and Path(interproscan_script).resolve() == INTERPROSCAN_SCRIPT.resolve()
        ):
            activated, activation_msg = activate_bundled_interproscan_binaries(SCRIPT_DIR)
            print(activation_msg)
            if not activated:
                return False

        java_executable = resolve_java_executable(SCRIPT_DIR) if resolve_java_executable else None
        if java_executable is None:
            print("Error: Java runtime not found. Install Java 11+ or place a JDK under .local-jdk/")
            return False
             
        cmd = [interproscan_script, '-i', fasta_file, '-f', 'json', '-d', str(ipr_output_dir)]
        
        if appl_list:
            cmd.extend(['-appl', appl_list])
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Execute
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=build_runtime_env(SCRIPT_DIR) if build_runtime_env else None,
        )
        
        if result.returncode == 0:
            print("InterProScan completed")
            return True
        else:
            print(f"InterProScan failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Error while running InterProScan: {e}")
        return False

# Run hmmscan
def run_hmmscan(fasta_file, output_dir, interproscan_path=None):
    # Ensure db directory is available
    if not interproscan_path and not ensure_db_extracted():
        print("Error: db directory is not available")
        return False
        
    # Set hmmscan-related paths
    hmm_db = SCRIPT_DIR / "hmm" / "self_build.hmm"
    
    if not hmm_db.exists():
        print(f"Error: HMM database file not found: {hmm_db}")
        return False
    
    if not Path(fasta_file).exists():
        print(f"Error: FASTA file not found: {fasta_file}")
        return False
    
    # Prefer bundled hmmscan (db/interproscan/bin/hmmer/hmmer3/hmmscan)
    
    hmmscan_executable = None

    if resolve_helper_executable is not None:
        helper_hmmscan = resolve_helper_executable(SCRIPT_DIR, "hmmer3", "hmmscan")
        if helper_hmmscan is not None:
            hmmscan_executable = str(helper_hmmscan)
            print(f"Using platform helper hmmscan: {hmmscan_executable}")
    
    if interproscan_path:
        # Try hmmscan bundled with a user-specified InterProScan
        interpro_dir = Path(interproscan_path).parent
        candidates = []
        candidates.append(interpro_dir / "bin" / "hmmer" / "hmmer3" / "hmmscan")
        candidates.append(interpro_dir / "bin" / "hmmer" / "hmmscan")
        candidates.append(interpro_dir / "bin" / "hmmscan")
        hmmer3_dir = interpro_dir / "bin" / "hmmer" / "hmmer3"
        if hmmer3_dir.exists():
            for p in hmmer3_dir.rglob("hmmscan"):
                candidates.append(p)
        bin_dir = interpro_dir / "bin"
        if bin_dir.exists():
            for p in bin_dir.rglob("hmmscan"):
                candidates.append(p)

        for c in candidates:
            try:
                if c.exists() and os.access(c, os.X_OK) and c.is_file():
                    hmmscan_executable = str(c)
                    print(f"Using bundled hmmscan from custom InterProScan: {hmmscan_executable}")
                    break
            except Exception:
                pass
        # If not found, prefer system hmmscan if available
        if not hmmscan_executable and shutil.which("hmmscan"):
            hmmscan_executable = "hmmscan"
            print("Using system hmmscan")
            
    if not hmmscan_executable:
        internal_hmmscan = DB_DIR / "interproscan" / "bin" / "hmmer" / "hmmer3" / "hmmscan"
        if internal_hmmscan.exists() and os.access(internal_hmmscan, os.X_OK):
            hmmscan_executable = str(internal_hmmscan)
            print(f"Using bundled hmmscan: {hmmscan_executable}")
        else:
            hmmscan_executable = "hmmscan"
            if shutil.which("hmmscan") is None:
                print("Error: hmmscan executable not found (neither bundled nor on system PATH)")
                return False
            print(f"Using system hmmscan: {hmmscan_executable}")

    try:
        # Create hmmscan output directory
        hmmscan_output_dir = Path(output_dir) / "hmmscan"
        hmmscan_output_dir.mkdir(exist_ok=True)
        
        # Set output file path
        output_file = hmmscan_output_dir / "result.tbl"
        
        # Build hmmscan command
        cmd = [
            hmmscan_executable,
            "--tblout", str(output_file),
            "--noali",
            str(hmm_db),
            str(fasta_file)
        ]
        
        print(f"Running hmmscan command: {' '.join(cmd)}")
        
        # Execute hmmscan
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=build_runtime_env(SCRIPT_DIR) if build_runtime_env else None,
        )
        
        print("hmmscan completed successfully")
        print(f"Results saved to: {output_file}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"hmmscan failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error while running hmmscan: {e}")
        return False


def preprocess_input_sequences(context):
    return _ensure_processed_fasta(context)


def run_tf_prediction_step(
    context,
    threshold,
    predict_mode="fast",
    debug=False,
    use_supplementary=False,
    supplementary_only=False,
    supp_models=None,
    step_number=1,
):
    print(f"{_format_step_heading(step_number, 'Running TF prediction')}...")

    context.preclass_dir.mkdir(exist_ok=True)
    cmd = _build_prediction_command(
        fasta_file=context.input_fasta,
        threshold=threshold,
        output_csv=context.prediction_csv,
        project_output=context.project_output,
        predict_mode=predict_mode,
        debug=debug,
        use_supplementary=use_supplementary,
        supplementary_only=supplementary_only,
        supp_models=supp_models,
    )

    print(f"Executing command: {' '.join(cmd)}")
    step_start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=SCRIPT_DIR,
        env=build_runtime_env(SCRIPT_DIR) if build_runtime_env else None,
    )
    step_duration = time.time() - step_start_time

    if result.returncode != 0:
        print(f"Prediction failed: {result.stderr}")
        return False

    print(f"Prediction completed (elapsed: {format_duration(step_duration)})")
    if result.stdout:
        print(result.stdout)

    return context.prediction_csv


def extract_predicted_tf_sequences_step(context, processed_fasta, threshold, step_number=2):
    print(f"{_format_step_heading(step_number, 'Extracting predicted TF sequences')}...")

    context.preclass_dir.mkdir(exist_ok=True)
    prediction_file = context.prediction_csv
    if not prediction_file.exists():
        print(f"Warning: prediction result file not found: {prediction_file}")
        return None

    print(f"Using prediction result file: {prediction_file}")
    step_start_time = time.time()
    tf_fasta = extract_tf_sequences_from_csv(
        str(processed_fasta),
        str(prediction_file),
        str(context.preclass_dir),
        threshold,
    )
    step_duration = time.time() - step_start_time

    if tf_fasta is None:
        print("No TF sequences met the prediction threshold")
        return None

    print(f"TF sequences saved to: {tf_fasta} (elapsed: {format_duration(step_duration)})")
    return Path(tf_fasta)


def run_protein_kinase_step(context, fasta_file, enabled=True, debug=False, step_number=None):
    print(f"{_format_step_heading(step_number, 'Running protein kinase analysis')}...")

    step_start_time = time.time()
    success = _run_protein_kinase_analysis(
        fasta_file,
        context.project_output,
        enabled=enabled,
        debug=debug,
    )
    step_duration = time.time() - step_start_time

    if success:
        print(f"Protein kinase step completed (elapsed: {format_duration(step_duration)})")

    return success


def run_interproscan_step(context, fasta_file, appl_list=None, interproscan_path=None, step_number=None):
    print(f"{_format_step_heading(step_number, 'Running InterProScan')}...")

    step_start_time = time.time()
    success = run_interproscan(str(fasta_file), str(context.project_output), appl_list, interproscan_path)
    step_duration = time.time() - step_start_time

    if success:
        print(f"InterProScan completed (elapsed: {format_duration(step_duration)})")
    else:
        print("InterProScan failed")

    return success


def run_hmmscan_step(context, fasta_file, interproscan_path=None, step_number=None):
    print(f"{_format_step_heading(step_number, 'Running hmmscan')}...")

    step_start_time = time.time()
    success = run_hmmscan(str(fasta_file), str(context.project_output), interproscan_path)
    step_duration = time.time() - step_start_time

    if success:
        print(f"hmmscan completed (elapsed: {format_duration(step_duration)})")
    else:
        print("hmmscan failed")

    return success


def run_tf_classification_step(
    context,
    use_predicted=True,
    debug=False,
    score_threshold=1.0,
    classification_mode="specific",
    step_number=None,
):
    print(f"{_format_step_heading(step_number, 'Running classification analysis')}...")

    step_start_time = time.time()
    success = run_analysis_modules(
        context,
        use_predicted=use_predicted,
        debug=debug,
        score_threshold=score_threshold,
        classification_mode=classification_mode,
    )
    step_duration = time.time() - step_start_time

    if success:
        print(f"Analysis modules completed (elapsed: {format_duration(step_duration)})")
    else:
        print("Analysis modules failed")

    return success


def run_tf_tr_analysis_pipeline(
    context,
    analysis_fasta,
    use_predicted,
    appl_list=None,
    debug=False,
    score_threshold=1.0,
    classification_mode="specific",
    interproscan_path=None,
    run_interproscan_analysis=True,
    run_hmmscan_analysis=True,
    step_numbers=(None, None, None),
):
    interproscan_step, hmmscan_step, classification_step = step_numbers

    if run_interproscan_analysis:
        if not run_interproscan_step(
            context,
            analysis_fasta,
            appl_list=appl_list,
            interproscan_path=interproscan_path,
            step_number=interproscan_step,
        ):
            return False

    if run_hmmscan_analysis:
        if not run_hmmscan_step(
            context,
            analysis_fasta,
            interproscan_path=interproscan_path,
            step_number=hmmscan_step,
        ):
            return False

    if run_interproscan_analysis and run_hmmscan_analysis:
        return run_tf_classification_step(
            context,
            use_predicted=use_predicted,
            debug=debug,
            score_threshold=score_threshold,
            classification_mode=classification_mode,
            step_number=classification_step,
        )

    return True


def write_empty_tf_tr_outputs_step(context, debug=False, step_number=None):
    print(f"{_format_step_heading(step_number, 'Writing empty TF/TR outputs')}...")

    step_start_time = time.time()
    context.preclass_dir.mkdir(exist_ok=True)
    context.result_dir.mkdir(exist_ok=True)

    if not context.tf_fasta.exists():
        context.tf_fasta.write_text("", encoding="utf-8")

    if write_tftr_outputs is not None:
        written_outputs = write_tftr_outputs({}, context.result_dir, debug=debug)
        match_tbl_path = written_outputs["table"]
    else:
        match_tbl_path = context.result_dir / "match_tbl.txt"
        match_tbl_path.write_text("", encoding="utf-8")

    classified_fasta_path = context.result_dir / f"{context.tf_fasta.stem}_tf_classified.fasta"
    classified_fasta_path.write_text("", encoding="utf-8")

    if debug and write_tftr_outputs is None:
        match_json_path = context.result_dir / "match.json"
        with open(match_json_path, "w", encoding="utf-8") as handle:
            json.dump({}, handle, indent=2, ensure_ascii=False)

    _write_combined_result_summary(context.project_output)
    step_duration = time.time() - step_start_time

    print("No TF sequences met the prediction threshold; wrote empty TF/TR outputs")
    print(f"TF FASTA: {context.tf_fasta}")
    print(f"Classification table: {match_tbl_path}")
    print(f"Classified FASTA: {classified_fasta_path}")
    if debug:
        print(f"Classification JSON: {context.result_dir / 'match.json'}")
    print(f"Empty TF/TR finalization completed (elapsed: {format_duration(step_duration)})")

    return True

# ============================================================================
# Main functional entry points
# ============================================================================
# Full prediction workflow
def predict_transcription_factors(threshold, fasta_file, output=None, extract_sequences=True, 
                                run_interproscan_analysis=True, run_hmmscan_analysis=True, 
                                appl_list=None, debug=False, score_threshold=1.0, classification_mode='specific',
                                predict_mode='fast', grad_cam_mode='none', interproscan_path=None,
                                use_supplementary=False, supplementary_only=False, supp_models=None,
                                run_protein_kinase_analysis=True):
    # Record start time
    predict_start_time = time.time()
    
    try:
        # Input validation
        if not Path(fasta_file).exists():
            print(f"Error: input file not found: {fasta_file}")
            return False
        
        # Set project output directory
        context = build_pipeline_context(fasta_file, output=output)
        project_output = context.project_output
        
        print("\n=== Starting TF Prediction ===\n")
        print(f"Input file: {fasta_file}")
        print(f"Output directory: {project_output}")
        print(f"Prediction threshold: {threshold}")
        print(f"Prediction mode: {predict_mode}")
        if grad_cam_mode != 'none':
            print(f"Grad-CAM: enabled (mode: {grad_cam_mode})")

        processed_fasta = preprocess_input_sequences(context)
        prediction_csv = run_tf_prediction_step(
            context,
            threshold=threshold,
            predict_mode=predict_mode,
            debug=debug,
            use_supplementary=use_supplementary,
            supplementary_only=supplementary_only,
            supp_models=supp_models,
            step_number=1,
        )
        if not prediction_csv:
            return False

        tf_fasta = None
        if extract_sequences:
            tf_fasta = extract_predicted_tf_sequences_step(
                context,
                processed_fasta=processed_fasta,
                threshold=threshold,
                step_number=2,
            )
        
        pk_success = run_protein_kinase_step(
            context,
            processed_fasta,
            enabled=run_protein_kinase_analysis,
            debug=debug,
            step_number=3,
        )
        if not pk_success:
            return False

        if tf_fasta is None:
            if not write_empty_tf_tr_outputs_step(context, debug=debug, step_number=4):
                return False

            predict_end_time = time.time()
            total_predict_duration = predict_end_time - predict_start_time

            print("\n=== TF Prediction Completed ===")
            print(f"Total runtime: {format_duration(total_predict_duration)}")
            print()
            return True

        if tf_fasta:
            analysis_success = run_tf_tr_analysis_pipeline(
                context,
                analysis_fasta=tf_fasta,
                use_predicted=True,
                appl_list=appl_list,
                debug=debug,
                score_threshold=score_threshold,
                classification_mode=classification_mode,
                interproscan_path=interproscan_path,
                run_interproscan_analysis=run_interproscan_analysis,
                run_hmmscan_analysis=run_hmmscan_analysis,
                step_numbers=(4, 5, 6),
            )
            if not analysis_success:
                return False
        
        # 7. Generate Grad-CAM heatmaps
        if grad_cam_mode != 'none':
            print(f"\n7. Generating Grad-CAM heatmaps (mode: {grad_cam_mode})...")
            step_start_time = time.time()
            
            result_dir = project_output / "result"
            input_stem = Path(fasta_file).stem

            target_fasta = None
            if grad_cam_mode == 'fast':
                candidates = list(result_dir.glob(f"{input_stem}*_tf_classified.fasta"))
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    target_fasta = str(candidates[0])
                else:
                    print(f"Classified FASTA not found under result/: {result_dir}; skipping Grad-CAM")
            elif grad_cam_mode == 'all':
                target_fasta = str(Path(fasta_file))
            else:
                target_fasta = str(Path(fasta_file))
            
            if target_fasta:
                # Temporary output file for Grad-CAM pass
                temp_pred_csv = context.preclass_dir / f"{input_stem}_gradcam_prediction.csv"
                grad_cam_cmd = _build_prediction_command(
                    fasta_file=target_fasta,
                    threshold=threshold,
                    output_csv=temp_pred_csv,
                    project_output=project_output,
                    predict_mode=predict_mode,
                    debug=debug,
                    use_supplementary=use_supplementary,
                    supplementary_only=supplementary_only,
                    supp_models=supp_models,
                    grad_cam_mode=grad_cam_mode,
                )
                
                print("Executing Grad-CAM generation...")
                gc_result = subprocess.run(
                    grad_cam_cmd,
                    capture_output=True,
                    text=True,
                    cwd=SCRIPT_DIR,
                    env=build_runtime_env(SCRIPT_DIR) if build_runtime_env else None,
                )
                
                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                
                if gc_result.returncode == 0:
                    print(f"Grad-CAM completed (elapsed: {format_duration(step_duration)})")
                else:
                    print(f"Grad-CAM failed: {gc_result.stderr}")
            else:
                step_end_time = time.time()
                step_duration = step_end_time - step_start_time
                print(f"Grad-CAM skipped (elapsed: {format_duration(step_duration)})")
            
            cleanup_processed_fasta(project_output, fasta_file)

        # Total time
        predict_end_time = time.time()
        total_predict_duration = predict_end_time - predict_start_time
        
        print("\n=== TF Prediction Completed ===")
        print(f"Total runtime: {format_duration(total_predict_duration)}")
        print()
        return True
        
    except Exception as e:
        print(f"Error during prediction workflow: {e}")
        return False
# Direct analysis mode (no prediction)
def analyze_sequences_directly(fasta_file, output=None, appl_list=None, debug=False, score_threshold=1.0, classification_mode='specific', interproscan_path=None, run_protein_kinase_analysis=True):
    # Record start time
    analysis_start_time = time.time()
    
    try:
        # Input validation
        if not Path(fasta_file).exists():
            print(f"Error: input file not found: {fasta_file}")
            return False
        
        # Set project output directory
        context = build_pipeline_context(fasta_file, output=output)
        project_output = context.project_output
        
        print("\n=== Starting Sequence Analysis ===\n")
        print(f"Input file: {fasta_file}")
        print(f"Output directory: {project_output}")
        
        processed_fasta = preprocess_input_sequences(context)

        pk_success = run_protein_kinase_step(
            context,
            processed_fasta,
            enabled=run_protein_kinase_analysis,
            debug=debug,
            step_number=1,
        )
        if not pk_success:
            return False

        analysis_success = run_tf_tr_analysis_pipeline(
            context,
            analysis_fasta=processed_fasta,
            use_predicted=False,
            appl_list=appl_list,
            debug=debug,
            score_threshold=score_threshold,
            classification_mode=classification_mode,
            interproscan_path=interproscan_path,
            run_interproscan_analysis=True,
            run_hmmscan_analysis=True,
            step_numbers=(2, 3, 4),
        )
        if not analysis_success:
            return False
        
        # Total time
        analysis_end_time = time.time()
        total_analysis_duration = analysis_end_time - analysis_start_time
        
        print("\n=== Sequence Analysis Completed ===")
        print(f"Total runtime: {format_duration(total_analysis_duration)}")
        print()
        return True
        
    except Exception as e:
        print(f"Error during analysis workflow: {e}")
        return False

# Test mode: reuse existing results to validate classification
def run_test_mode(fasta_file, json_file, spechmm_file, output=None, debug=False, score_threshold=1.0, classification_mode='specific'):
    print(f"\n{'='*50}")
    print("Starting test mode")
    print("Skipping InterProScan and hmmscan; using provided files for classification validation")
    print(f"{'='*50}")
    
    try:
        # Set project output directory
        context = build_pipeline_context(fasta_file, output=output)
        project_output = context.project_output
        print(f"Project output directory: {project_output}")
        
        # Create required subdirectories
        ipr_dir = context.interproscan_dir
        hmmscan_dir = context.hmmscan_dir
        result_dir = context.result_dir
        
        ipr_dir.mkdir(exist_ok=True)
        hmmscan_dir.mkdir(exist_ok=True)
        result_dir.mkdir(exist_ok=True)
        
        print("\nCreating project directory structure:")
        print(f"  IPR dir: {ipr_dir}")
        print(f"  hmmscan dir: {hmmscan_dir}")
        print(f"  Result dir: {result_dir}")
        
        # Copy provided files into the project directory
        input_filename = Path(fasta_file).name
        target_json_file = ipr_dir / f"{input_filename}.json"
        target_hmmscan_file = hmmscan_dir / "result.tbl"
        
        print("\nCopying inputs into project directory:")
        print(f"  Copy {json_file} -> {target_json_file}")
        shutil.copy2(json_file, target_json_file)
        
        print(f"  Copy {spechmm_file} -> {target_hmmscan_file}")
        shutil.copy2(spechmm_file, target_hmmscan_file)
        
        # Verify copied files
        if not target_json_file.exists():
            print(f"Error: failed to copy JSON file: {target_json_file}")
            return False
        
        if not target_hmmscan_file.exists():
            print(f"Error: failed to copy hmmscan file: {target_hmmscan_file}")
            return False
        
        print("\nCopy complete; starting analysis modules...")
        
        # Record start time
        analysis_start_time = time.time()
        
        # Run analysis modules (use original FASTA; no prediction)
        analysis_success = run_analysis_modules(context, use_predicted=False, debug=debug, score_threshold=score_threshold, classification_mode=classification_mode)
        
        # Compute runtime
        analysis_end_time = time.time()
        analysis_duration = analysis_end_time - analysis_start_time
        
        if analysis_success:
            print(f"\n{'='*50}")
            print("Test-mode analysis completed")
            print(f"Runtime: {format_duration(analysis_duration)}")
            print(f"Results saved to: {result_dir}")
            print(f"{'='*50}")
            return True
        else:
            print(f"\n{'='*50}")
            print("Test-mode analysis failed")
            print(f"Runtime: {format_duration(analysis_duration)}")
            print(f"{'='*50}")
            return False
        
    except Exception as e:
        print(f"Error during test-mode execution: {e}")
        return False

# ============================================================================
# Argument validators
# ============================================================================

def validate_appl_list(appl_string):
    valid_apps = {'CDD', 'NCBIfam', 'PANTHER', 'Pfam', 'PROSITEPATTERNS', 'PROSITEPROFILES', 'SMART'}
    
    if appl_string:
        apps = [app.strip() for app in appl_string.split(',')]
        invalid_apps = [app for app in apps if app not in valid_apps]
        
        if invalid_apps:
            raise argparse.ArgumentTypeError(f"Invalid application(s): {', '.join(invalid_apps)}. Allowed: {', '.join(sorted(valid_apps))}")
    
    return appl_string
# Validate prediction threshold in [0, 1]
def validate_threshold(value):
    try:
        threshold = float(value)
        if not 0 <= threshold <= 1:
            raise argparse.ArgumentTypeError("Threshold must be between 0 and 1")
        return threshold
    except ValueError:
        raise argparse.ArgumentTypeError("Threshold must be a number")

# ============================================================================
# Main
# ============================================================================

def main():
    # Record start time
    program_start_time = time.time()
    start_datetime = datetime.now()
    print("\n=== iTAK3 Started ===")
    print(f"Start time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"="*50)
    
    parser = argparse.ArgumentParser(
        description="iTAK3 - transcription factor prediction and analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # 1) Analyze input sequences directly (skip prediction)
  python itak3-v1.0.py -i input.fasta
  
  # 2) Enable prediction: predict TFs, then run domain analysis and family classification
  python itak3-v1.0.py --predict -i input.fasta -t 0.1
  
  # 3) Specify an output directory (default: output/<input_basename>/)
  python itak3-v1.0.py --predict -i input.fasta -o /path/to/output
  
  # 4) Restrict InterProScan applications (run only selected libraries)
  python itak3-v1.0.py -i input.fasta --appl CDD,Pfam,SMART
  
  # 5) Adjust InterProScan score filtering (retain only hits with score >= threshold)
  python itak3-v1.0.py -i input.fasta --score 1.0
  
  # 6) Classification policy: specific (specificity-first) or score (score-first, default)
  python itak3-v1.0.py -i input.fasta --classification-mode specific
  
  # 7) Prediction splitting mode: fast (default) or full (more comprehensive, slower)
  python itak3-v1.0.py --predict -i input.fasta --predict-mode full
  
  # 8) Use supplementary models (stacked with main model, or supplementary-only)
  python itak3-v1.0.py --predict --use-supplementary -i input.fasta
  python itak3-v1.0.py --predict --supplementary-only -i input.fasta
  python itak3-v1.0.py --predict --use-supplementary --supp-models a.pth b.pth -i input.fasta
  
  # 9) Grad-CAM heatmaps (available only with --predict; fast runs after classification in batch)
  python itak3-v1.0.py --predict --grad-cam-mode fast -i input.fasta
  python itak3-v1.0.py --predict --grad-cam-mode all -i input.fasta
  
  # 10) Test mode: skip InterProScan/hmmscan and validate classification using existing files
  python itak3-v1.0.py -test -i input.fasta -json ipr_result.json -spechmm hmmscan_result.tbl
  
  # 11) Use an external interproscan.sh (hmmscan will follow the specified InterProScan)
  python itak3-v1.0.py -i input.fasta --interproscan /path/to/interproscan.sh
  
  # 12) Check dependencies and exit
  python itak3-v1.0.py --check-deps
        """
    )
    
    # Primary parameters
    parser.add_argument('-i', '--input', help='Path to input FASTA file')
    parser.add_argument('-t', '--threshold', type=validate_threshold, default=0.1, 
                       help='Prediction threshold in [0, 1] (default: 0.1)')
    parser.add_argument('-o', '--output', help='Output directory path')
    
    # Feature options
    parser.add_argument('--predict', action='store_true', 
                       help='Enable prediction (model inference is used as downstream analysis input)')
    parser.add_argument('--list-predict', action='store_true',
                       help='Predict once, then classify under multiple thresholds to produce multiple outputs')
    # parser.add_argument('--extract-sequences', action='store_true', default=True, 
    #                    help='Extract predicted TF sequences (only with --predict)')
    
    # Test mode parameters
    parser.add_argument('-test', '--test-mode', action='store_true',
                       help='Enable test mode (skip InterProScan/hmmscan and validate classification using specified files)')
    parser.add_argument('-json', '--json-file', 
                       help='InterProScan JSON result file path for test mode (only with -test)')
    parser.add_argument('-spechmm', '--spechmm-file',
                       help='hmmscan result file path for test mode (only with -test)')
    
    # InterProScan applications
    parser.add_argument('--appl', type=validate_appl_list,
                       default='CDD,PANTHER,Pfam,PROSITEPATTERNS,PROSITEPROFILES,SMART',
                       help='InterProScan application list (comma-separated). Allowed: CDD,NCBIfam,PANTHER,Pfam,PROSITEPATTERNS,PROSITEPROFILES,SMART')
    
    # External InterProScan path
    parser.add_argument('--interproscan', help='Path to interproscan.sh (hmmscan will follow the specified InterProScan)')
    parser.add_argument('--skip-pk', action='store_true',
                       help='Skip protein kinase identification/classification (enabled by default in direct/predict modes)')

    # Debug mode
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug mode and write intermediate artifacts (default: off)')
    
    # Score threshold
    parser.add_argument('--score', type=float, default=1.0,
                       help='Score threshold for InterProScan filtering; retain only hits above this value (default: 1.0)')
    
    # Classification mode
    parser.add_argument('--classification-mode', choices=['specific', 'score'], default='score',
                       help="Classification mode: 'specific' (specificity-first) or 'score' (score-first, default)")
    
    # Prediction splitting mode
    parser.add_argument('--predict-mode', choices=['fast', 'full'], default='fast',
                       help="Prediction splitting mode: 'fast' (default) or 'full' (full coverage)")
    # Supplementary model options
    parser.add_argument('--use-supplementary', action='store_true',
                       help='Enable supplementary models for an additional prediction pass (default: off)')
    parser.add_argument('--supplementary-only', action='store_true',
                       help='Use supplementary models only; skip the main model')
    parser.add_argument('--supp-models', nargs='*', default=None,
                       help='Optional: specify supplementary model path(s); if omitted, use all models in the directory')
    
    # Grad-CAM options
    parser.add_argument('--grad-cam-mode', choices=['none', 'all', 'positive', 'fast'], default='none',
                       help='Grad-CAM mode: none (default), all, positive (thresholded), fast')

    # Dependency checks
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies and exit')
    parser.add_argument('--skip-deps-check', action='store_true',
                       help='Skip dependency checks and run anyway (not recommended)')
    
    args = parser.parse_args()
    
    # Check dependencies and exit
    if args.check_deps:
        if DependencyChecker:
            checker = DependencyChecker(interproscan_path=args.interproscan)
            success = checker.run_full_check()
            sys.exit(0 if success else 1)
        else:
            print("Error: dependency checker module is unavailable")
            sys.exit(1)
    
    # Run dependency checks (unless explicitly skipped)
    if not args.skip_deps_check and DependencyChecker:
        print(" Checking runtime dependencies...")
        checker = DependencyChecker(interproscan_path=args.interproscan)
        
        # Decide whether to check prediction dependencies
        check_predict = args.predict or args.list_predict
        dependencies_ok = checker.run_full_check(
            check_predict=check_predict,
            skip_db_check=args.test_mode,
            test_mode=args.test_mode,
        )
        
        # If db assets are missing, run_full_check attempts a download/extraction.
        
        if not dependencies_ok and not args.test_mode and not args.interproscan:
            # Re-check db readiness in case it was prepared during the previous run
            if checker.ensure_db_extracted():
                 print("\n[INFO] db assets are ready; re-checking all dependencies...")
                 dependencies_ok = checker.run_full_check(
                     check_predict=check_predict,
                     skip_db_check=args.test_mode,
                     test_mode=args.test_mode,
                 )

        if not dependencies_ok:
            print("\n[ERROR] Dependency checks failed. The program may not run correctly.")
            print("Install the missing dependencies, or use --skip-deps-check to force execution (not recommended).")
            print("Use --check-deps to run dependency checks only.")
            sys.exit(1)
        
        print("[OK] Dependency checks passed\n")
    elif args.skip_deps_check:
        print("[WARN] Dependency checks were skipped\n")
    
    # Validate input argument
    if not args.input:
        print("Error: input file must be provided (-i/--input)")
        sys.exit(1)
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)
    
    # Validate FASTA format
    if FastaValidator:
        print(" Validating FASTA format...")
        validator = FastaValidator()
        is_valid = validator.run_full_validation(args.input)
        
        if not is_valid:
            print("[ERROR] FASTA validation failed. Review the errors above.")
            sys.exit(1)
        
        print("[OK] FASTA validation passed\n")
    else:
        print("[WARN] FASTA validation module unavailable; skipping format validation\n")
    
    # Grad-CAM argument validation
    if args.grad_cam_mode in ['all', 'positive'] and not (args.predict or args.list_predict):
        print("Error: Grad-CAM modes 'all' and 'positive' require --predict")
        sys.exit(1)
    if args.list_predict and args.grad_cam_mode != 'none':
        print("Error: --list-predict does not currently support Grad-CAM; use --grad-cam-mode none")
        sys.exit(1)
        
    # Validate test mode arguments
    if args.test_mode:
        # In test mode, prediction is not allowed
        if args.predict:
            print("Error: --predict cannot be used in test mode (-test)")
            sys.exit(1)
        if args.list_predict:
            print("Error: --list-predict cannot be used in test mode (-test)")
            sys.exit(1)
        
        # Test mode requires json and spechmm files
        if not args.json_file:
            print("Error: test mode requires -json (InterProScan JSON result file)")
            sys.exit(1)
        
        if not args.spechmm_file:
            print("Error: test mode requires -spechmm (hmmscan result file)")
            sys.exit(1)
        
        # Validate test-mode files exist
        if not Path(args.json_file).exists():
            print(f"Error: InterProScan JSON file not found: {args.json_file}")
            sys.exit(1)
        
        if not Path(args.spechmm_file).exists():
            print(f"Error: hmmscan result file not found: {args.spechmm_file}")
            sys.exit(1)
    
    # Disallow test-only arguments outside test mode
    if not args.test_mode:
        if args.json_file:
            print("Error: -json can be used only in test mode (-test)")
            sys.exit(1)
        
        if args.spechmm_file:
            print("Error: -spechmm can be used only in test mode (-test)")
            sys.exit(1)
    
    # Dispatch by mode
    if args.test_mode:
        print("Using test mode")
        success = run_test_mode(
            fasta_file=args.input,
            json_file=args.json_file,
            spechmm_file=args.spechmm_file,
            output=args.output,
            debug=args.debug,
            score_threshold=args.score,
            classification_mode=args.classification_mode
        )
        if success and args.grad_cam_mode != 'none':
            try:
                print(f"\n=== Running Grad-CAM Visualization ({args.grad_cam_mode} Mode) ===")
                project_output = resolve_existing_project_output(args.input, args.output)
                result_dir = project_output / "result"
                fasta_stem = Path(args.input).stem

                target_fasta = None
                if args.grad_cam_mode == 'fast':
                    candidates = list(result_dir.glob(f"{fasta_stem}*_tf_classified.fasta"))
                    if candidates:
                        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        target_fasta = str(candidates[0])
                        print(f"Using result FASTA for fast Grad-CAM: {target_fasta}")
                    else:
                        print(f"Classified FASTA not found under result/: {result_dir}; skipping Grad-CAM")
                        target_fasta = None
                else:
                    target_fasta = str(args.input)
                    print(f"Using original input FASTA for Grad-CAM: {target_fasta}")

                preclass_dir = project_output / "protein_model_preclassification"
                preclass_dir.mkdir(exist_ok=True)
                output_file = preclass_dir / f"{fasta_stem}_prediction.csv"

                if target_fasta:
                    cmd = [
                        sys.executable, str(PREDICT_SCRIPT),
                        "--fasta", str(target_fasta),
                        "--threshold", str(args.threshold),
                        "--output", str(output_file),
                        "--project-output", str(project_output),
                        "--mode", args.predict_mode,
                        "--grad-cam-mode", str(args.grad_cam_mode)
                    ]
                    if args.debug:
                        cmd.append("--debug")
                    print(f"Executing Grad-CAM command: {' '.join(cmd)}")
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        cwd=SCRIPT_DIR,
                        env=build_runtime_env(SCRIPT_DIR) if build_runtime_env else None,
                    )
                    if result.returncode == 0:
                        print("Grad-CAM heatmaps generated successfully")
                    else:
                        print(f"Grad-CAM failed: {result.stderr}")

                cleanup_processed_fasta(project_output, args.input)
            except Exception as e:
                print(f"Error while running Grad-CAM: {e}")
    elif args.list_predict:
        print("Using List Predict mode (predict once, classify under multiple thresholds)")

        if DependencyChecker:
            checker = DependencyChecker(interproscan_path=args.interproscan)
            prediction_ok, error_msg = checker.check_prediction_dependencies()
            if not prediction_ok:
                print(f"\n[ERROR] Prediction dependency checks failed: {error_msg}")
                print("\nNote: --list-predict requires PyTorch and model files.")
                sys.exit(1)
            print("[OK] Prediction dependency checks passed")

        if args.predict:
            print("Error: --list-predict cannot be used together with --predict")
            sys.exit(1)

        success = list_predict_transcription_factors(
            fasta_file=args.input,
            output=args.output,
            appl_list=args.appl,
            debug=args.debug,
            score_threshold=args.score,
            classification_mode=args.classification_mode,
            predict_mode=args.predict_mode,
            interproscan_path=args.interproscan,
            use_supplementary=args.use_supplementary,
            supplementary_only=args.supplementary_only,
            supp_models=args.supp_models,
            run_protein_kinase_analysis=not args.skip_pk,
        )
    elif args.predict:
        # Prediction mode
        print(f"Using prediction mode; threshold: {args.threshold}")
        
        # Check prediction dependencies
        if DependencyChecker:
            checker = DependencyChecker(interproscan_path=args.interproscan)
            prediction_ok, error_msg = checker.check_prediction_dependencies()
            
            if not prediction_ok:
                print(f"\n[ERROR] Prediction dependency checks failed: {error_msg}")
                print("\nPrediction requires:")
                print("  - PyTorch")
                print("  - matplotlib")
                print("  - Model file (model.pth)")
                print("  - Prediction script (predict.py)")
                print("\nInstall PyTorch:")
                print("  # CPU:")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
                print("  # GPU (CUDA 11.8):")
                print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
                print("\nInstall plotting dependency:")
                print("  pip install matplotlib")
                print("  # More options: https://pytorch.org/get-started/locally/")
                print("\nTip: If prediction is not needed, remove --predict to run analysis directly.")
                sys.exit(1)
            
            print("[OK] Prediction dependency checks passed")
        
        success = predict_transcription_factors(
            threshold=args.threshold,
            fasta_file=args.input,
            output=args.output,
            extract_sequences=True,  # Must be True because this step is required
            run_interproscan_analysis=True,  # Enabled by default
            run_hmmscan_analysis=True,       # Enabled by default
            appl_list=args.appl,
            debug=args.debug,
            score_threshold=args.score,
            classification_mode=args.classification_mode,
            predict_mode=args.predict_mode,
            grad_cam_mode=args.grad_cam_mode,
            interproscan_path=args.interproscan,
            use_supplementary=args.use_supplementary,
            supplementary_only=args.supplementary_only,
            supp_models=args.supp_models,
            run_protein_kinase_analysis=not args.skip_pk,
        )
    else:
        # Direct analysis mode
        print("Using direct analysis mode (skipping prediction)")
        
        # Grad-CAM requires prediction dependencies
        if args.grad_cam_mode != 'none':
            print(f"Note: Grad-CAM is enabled (mode: {args.grad_cam_mode}); the model will be executed to generate heatmaps")
            if DependencyChecker:
                checker = DependencyChecker(interproscan_path=args.interproscan)
                prediction_ok, error_msg = checker.check_prediction_dependencies()
                
                if not prediction_ok:
                    print(f"\n[ERROR] Grad-CAM dependency checks failed: {error_msg}")
                    print("Install PyTorch and related assets, or disable Grad-CAM")
                    sys.exit(1)

        success = analyze_sequences_directly(
            fasta_file=args.input,
            output=args.output,
            appl_list=args.appl,
            debug=args.debug,
            score_threshold=args.score,
            classification_mode=args.classification_mode,
            interproscan_path=args.interproscan,
            run_protein_kinase_analysis=not args.skip_pk,
        )
        
        # If analysis succeeds and Grad-CAM is enabled, run predict.py to generate heatmaps
        if success and args.grad_cam_mode != 'none':
            try:
                print(f"\n=== Running Grad-CAM Visualization ({args.grad_cam_mode} Mode) ===")
                project_output = resolve_existing_project_output(args.input, args.output)
                
                result_dir = project_output / "result"
                fasta_stem = Path(args.input).stem

                target_fasta = None
                if args.grad_cam_mode == 'fast':
                    candidates = list(result_dir.glob(f"{fasta_stem}*_tf_classified.fasta"))
                    if candidates:
                        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        target_fasta = str(candidates[0])
                        print(f"Using result FASTA for fast Grad-CAM: {target_fasta}")
                    else:
                        print(f"Classified FASTA not found under result/: {result_dir}; skipping Grad-CAM")
                        target_fasta = None
                else:
                    target_fasta = str(args.input)
                    print(f"Using original input FASTA for Grad-CAM: {target_fasta}")
                
                # Create preclassification directory (required by predict.py)
                preclass_dir = project_output / "protein_model_preclassification"
                preclass_dir.mkdir(exist_ok=True)
                
                # Output filename
                output_file = preclass_dir / f"{fasta_stem}_prediction.csv"
                
                if target_fasta:
                    cmd = [
                        "python", str(PREDICT_SCRIPT),
                        "--fasta", str(target_fasta),
                        "--threshold", str(args.threshold),
                        "--output", str(output_file),
                        "--project-output", str(project_output),
                        "--mode", args.predict_mode,
                        "--grad-cam-mode", str(args.grad_cam_mode)
                    ]
                
                    if args.debug:
                        cmd.append("--debug")
                    
                    print(f"Executing Grad-CAM command: {' '.join(cmd)}")
                
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
                
                    if result.returncode == 0:
                        print("Grad-CAM heatmaps generated successfully")
                    else:
                        print(f"Grad-CAM failed: {result.stderr}")
                
                cleanup_processed_fasta(project_output, args.input)
                    
            except Exception as e:
                print(f"Error while running Grad-CAM: {e}")

    
    if success:
        print("\nAnalysis completed")
    else:
        print("\nAnalysis failed")
        # Compute and display runtime even on failure
        program_end_time = time.time()
        end_datetime = datetime.now()
        total_runtime = program_end_time - program_start_time
        
        print("\n=== iTAK3 Finished ===")
        print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total runtime: {format_duration(total_runtime)}")
        print(f"="*50)
        sys.exit(1)
    
    # Compute and display total runtime
    program_end_time = time.time()
    end_datetime = datetime.now()
    total_runtime = program_end_time - program_start_time
    
    print("\n=== iTAK3 Finished ===")
    print(f"End time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {format_duration(total_runtime)}")
    print(f"="*50)

if __name__ == '__main__':
    main()
