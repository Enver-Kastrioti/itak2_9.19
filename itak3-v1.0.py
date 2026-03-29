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


def _load_tf_tr_match_table(match_tbl_path):
    records = {}
    match_tbl_path = Path(match_tbl_path)
    if not match_tbl_path.exists():
        return records

    with open(match_tbl_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            seq_id, name, family, type_, desc, other_family = parts[:6]
            records[seq_id] = {
                "tftr_name": name,
                "tftr_family": family,
                "tftr_type": type_,
                "tftr_desc": desc,
                "tftr_other_family": other_family,
            }
    return records


def _load_pk_match_table(pk_tbl_path):
    records = {}
    pk_tbl_path = Path(pk_tbl_path)
    if not pk_tbl_path.exists():
        return records

    with open(pk_tbl_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            seq_id = row.get("Sequence_ID")
            if not seq_id:
                continue
            records[seq_id] = {
                "pk_shiu_class": row.get("Shiu_Class", "NA") or "NA",
                "pk_ppc_class": row.get("PPC_Class", "NA") or "NA",
                "pk_ppc_description": row.get("PPC_Description", "NA") or "NA",
            }
    return records


def _write_combined_result_summary(project_output):
    project_output = Path(project_output)
    result_dir = project_output / "result"
    result_dir.mkdir(exist_ok=True)

    tf_tr_records = _load_tf_tr_match_table(result_dir / "match_tbl.txt")
    pk_records = _load_pk_match_table(project_output / "protein_kinase" / "pk_classification.tsv")
    all_ids = sorted(set(tf_tr_records.keys()) | set(pk_records.keys()))

    summary_path = result_dir / "all_match_tbl.txt"
    with open(summary_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "Sequence_ID",
            "TFTR_Name",
            "TFTR_Family",
            "TFTR_Type",
            "TFTR_Description",
            "TFTR_Other_Family",
            "PK_Shiu_Class",
            "PK_PPC_Class",
            "PK_PPC_Description",
        ])
        for seq_id in all_ids:
            tf_tr = tf_tr_records.get(seq_id, {})
            pk = pk_records.get(seq_id, {})
            writer.writerow([
                seq_id,
                tf_tr.get("tftr_name", "NA"),
                tf_tr.get("tftr_family", "NA"),
                tf_tr.get("tftr_type", "NA"),
                tf_tr.get("tftr_desc", "NA"),
                tf_tr.get("tftr_other_family", "NA"),
                pk.get("pk_shiu_class", "NA"),
                pk.get("pk_ppc_class", "NA"),
                pk.get("pk_ppc_description", "NA"),
            ])

    print(
        "Combined result summary written: "
        f"{summary_path} ({len(all_ids)} sequences; "
        f"{len(tf_tr_records)} TF/TR, {len(pk_records)} PK)"
    )
    return True

def cleanup_processed_fasta(project_output, fasta_file):
    try:
        import importlib.util
        get_fasta_path = MODULE_DIR / "get_fasta.py"
        spec = importlib.util.spec_from_file_location("get_fasta", get_fasta_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        processed_fasta = Path(module.get_processed_fasta_path(fasta_file, project_output))
    except Exception:
        processed_fasta = Path(project_output) / f"{Path(fasta_file).stem}_protein_replaced.fasta"

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
def run_analysis_modules(project_output, fasta_file, use_predicted=True, debug=False, score_threshold=1.0, classification_mode='specific'):
    """
    Run analysis modules for transcription factor family classification.
    
    Args:
        project_output (Path): Project output directory.
        fasta_file (str): Input FASTA file path.
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
        # Determine which FASTA file to use
        if use_predicted:
            # Use predicted TF sequence FASTA
            tf_fasta_file = project_output / "protein_model_preclassification" / f"{Path(fasta_file).stem}_tf_sequences.fasta"
            if not tf_fasta_file.exists():
                candidates = sorted((project_output / "protein_model_preclassification").glob("*_tf_sequences.fasta"))
                if len(candidates) == 1:
                    tf_fasta_file = candidates[0]
                    print(f"Warning: predicted TF FASTA file not found: {project_output / 'protein_model_preclassification' / f'{Path(fasta_file).stem}_tf_sequences.fasta'}")
                    print(f"Using detected TF FASTA instead: {tf_fasta_file}")
                    analysis_fasta = str(tf_fasta_file)
                else:
                    print(f"Warning: predicted TF FASTA file not found: {tf_fasta_file}")
                    print("Falling back to the original input for analysis")
                    analysis_fasta = fasta_file
            else:
                analysis_fasta = str(tf_fasta_file)
        else:
            try:
                import importlib.util
                get_fasta_path = MODULE_DIR / "get_fasta.py"
                spec = importlib.util.spec_from_file_location("get_fasta", get_fasta_path)
                get_fasta = module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                processed_path = module.get_processed_fasta_path(fasta_file, project_output)
                if not Path(processed_path).exists():
                    module.generate_protein_fasta_with_translation(fasta_file, output_dir=project_output)
                analysis_fasta = processed_path
            except Exception:
                analysis_fasta = fasta_file
        
        # Determine other required file paths
        # InterProScan output filenames include the full input filename (including extension)
        input_filename = Path(analysis_fasta).name
        ipr_file = project_output / "InterproScan" / f"{input_filename}.json"
        hmmscan_file = project_output / "hmmscan" / "result.tbl"
        rule_file = SCRIPT_DIR / "rule.txt"

        # In test mode, ipr_file may need adjustment because run_test_mode may copy using a different name.
        if not ipr_file.exists():
            # Attempt to locate alternative JSON files
            # 1) Try JSON derived from the original FASTA filename (when analysis_fasta is a processed FASTA)
            original_json_file = project_output / "InterproScan" / f"{Path(fasta_file).name}.json"
            if original_json_file.exists():
                print(f"Warning: {ipr_file.name} not found; using {original_json_file.name} instead")
                ipr_file = original_json_file
            else:
                # 2) If there is exactly one JSON file in the directory, use it
                json_files = list((project_output / "InterproScan").glob("*.json"))
                if len(json_files) == 1:
                    print(f"Warning: {ipr_file.name} not found; using the only JSON file in directory: {json_files[0].name}")
                    ipr_file = json_files[0]
        
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
        print(f"  FASTA: {analysis_fasta}")
        print(f"  IPR JSON: {ipr_file}")
        print(f"  hmmscan: {hmmscan_file}")
        print(f"  Rule file: {rule_file}")
        
        # Create result output directory (under the project directory)
        result_dir = project_output / "result"
        result_dir.mkdir(exist_ok=True)
        print(f"Results will be written to: {result_dir}")
        
        # Step 0: run get_rule and write rule JSON under the project directory
        print("\nStep 0: running get_rule...")
        getrule_success = run_getrule_module(str(rule_file), str(project_output), debug=debug)
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
        class_tf_success = run_class_tf_module(
            analysis_fasta, 
            str(rule_file), 
            result_dir, 
            debug=debug,
            filtered_data=filtered_data,
            spec_data=spec_data,
            classification_mode=classification_mode
        )
        if not class_tf_success:
            print("classification failed")
            return False

        _write_combined_result_summary(project_output)
        
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
                # Write table output
                tbl_path = os.path.join(result_dir, 'match_tbl.txt')
                with open(tbl_path, 'w', encoding='utf-8') as f:
                    for gene_id, data in classification_result.items():
                        desc_str = ';'.join(data['desc']) if data['desc'] else 'NA'
                        line = f"{gene_id}\t{data['name']}\t{data['family']}\t{data['type']}\t{desc_str}\t{data['other_family']}\n"
                        f.write(line)
                
                # Generate classified FASTA
                try:
                    # Dynamically import get_fasta
                    get_fasta_path = MODULE_DIR / "get_fasta.py"
                    spec = importlib.util.spec_from_file_location("get_fasta", get_fasta_path)
                    get_fasta = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(get_fasta)
                    
                    if classification_result:
                        classified_fasta_path = get_fasta.generate_classified_fasta(
                            fasta_file, 
                            classification_result, 
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
                
                # Write match.json only in debug mode
                if debug:
                    json_path = os.path.join(result_dir, 'match.json')
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(classification_result, f, indent=2, ensure_ascii=False)
                    
                    print("classification completed (in-memory)")
                    print("Results saved to:")
                    print(f"  JSON: {json_path}")
                    print(f"  Table: {tbl_path}")
                else:
                    print("classification completed (in-memory)")
                    print("Results saved to:")
                    print(f"  Table: {tbl_path}")
                
                return True
            else:
                print("classification processing failed")
                return False
                
        except Exception as e:
            print(f"Error while running classification: {e}")
            return False
            
    except Exception as e:
        print(f"Error while running classification: {e}")
        return False

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
    try:
        import importlib.util
        get_fasta_path = MODULE_DIR / "get_fasta.py"
        spec = importlib.util.spec_from_file_location("get_fasta", get_fasta_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        processed_fasta = Path(module.get_processed_fasta_path(fasta_file, output_dir))
        if not processed_fasta.exists():
            module.generate_protein_fasta_with_translation(fasta_file, output_dir=output_dir)
        return str(processed_fasta)
    except Exception:
        return str(fasta_file)


def _run_protein_kinase_analysis(fasta_file, project_output, enabled=True):
    if not enabled:
        print("Protein kinase analysis skipped (--skip-pk)")
        return True

    result = call_module_function(
        "protein_kinase",
        "run_protein_kinase_pipeline",
        str(fasta_file),
        str(project_output),
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
    if not PREDICT_SCRIPT.exists():
        raise FileNotFoundError(f"Prediction script not found: {PREDICT_SCRIPT}")
    abs_fasta = str(Path(fasta_file).absolute())
    abs_project_output = str(Path(project_output).absolute())
    output_csv = str(Path(output_csv).absolute())

    cmd = [
        "python", str(PREDICT_SCRIPT),
        "--fasta", abs_fasta,
        "--threshold", str(threshold),
        "--output", output_csv,
        "--project-output", abs_project_output,
        "--mode", predict_mode,
    ]
    if supplementary_only:
        cmd.append("--supplementary-only")
    elif use_supplementary:
        cmd.append("--use-supplementary")
    if supp_models:
        cmd.append("--supp-models")
        for m in supp_models:
            cmd.append(m)
    if debug:
        cmd.append("--debug")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
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

    processed_fasta = _get_or_create_processed_fasta(fasta_file, cache_output)
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
        project_output = setup_project_output(fasta_file, str(child_output))
        preclass_dir = project_output / "protein_model_preclassification"
        preclass_dir.mkdir(exist_ok=True)

        prediction_csv = preclass_dir / f"{fasta_basename}_prediction.csv"
        tf_only_csv = preclass_dir / f"{fasta_basename}_prediction_tf_only.csv"
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

        print("\n3. Running InterProScan...")
        interproscan_success = run_interproscan(str(tf_fasta_path), str(project_output), appl_list, interproscan_path)
        if not interproscan_success:
            print("InterProScan failed")
            return False

        print("\n4. Running hmmscan...")
        hmmscan_success = run_hmmscan(str(tf_fasta_path), str(project_output), interproscan_path)
        if not hmmscan_success:
            print("hmmscan failed")
            return False

        print("\n5. Running classification analysis...")
        analysis_success = run_analysis_modules(
            project_output,
            fasta_file,
            use_predicted=True,
            debug=debug,
            score_threshold=score_threshold,
            classification_mode=classification_mode,
        )
        if not analysis_success:
            print("Analysis modules failed")
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
        project_output = setup_project_output(fasta_file, output)
        
        print("\n=== Starting TF Prediction ===\n")
        print(f"Input file: {fasta_file}")
        print(f"Output directory: {project_output}")
        print(f"Prediction threshold: {threshold}")
        print(f"Prediction mode: {predict_mode}")
        if grad_cam_mode != 'none':
            print(f"Grad-CAM: enabled (mode: {grad_cam_mode})")

        processed_fasta = _get_or_create_processed_fasta(fasta_file, project_output)
        
        # Build prediction command
        if not PREDICT_SCRIPT.exists():
            print(f"Error: prediction script not found: {PREDICT_SCRIPT}")
            return False
        
        # Output filename
        fasta_basename = Path(fasta_file).stem
        
        # Create preclassification output directory
        preclass_dir = project_output / "protein_model_preclassification"
        preclass_dir.mkdir(exist_ok=True)
        
        output_file = preclass_dir / f"{fasta_basename}_prediction.csv"
        
        abs_fasta = str(Path(fasta_file).absolute())
        abs_project_output = str(project_output.absolute())
        cmd = [
            sys.executable, str(PREDICT_SCRIPT),
            "--fasta", abs_fasta,
            "--threshold", str(threshold),
            "--output", str(output_file.absolute()),
            "--project-output", abs_project_output,
            "--mode", predict_mode
        ]
        if supplementary_only:
            cmd.append("--supplementary-only")
        elif use_supplementary:
            cmd.append("--use-supplementary")
        if supp_models:
            cmd.append("--supp-models")
            for m in supp_models:
                cmd.append(m)
        
        # Add Grad-CAM parameters
        # if grad_cam_mode != 'none':
        #    cmd.extend(["--grad-cam-mode", grad_cam_mode])
        
        # Add debug parameter
        if debug:
            cmd.append("--debug")
        
        print(f"Executing command: {' '.join(cmd)}")
        
        # Run prediction
        print("1. Running TF prediction...")
        step_start_time = time.time()
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=SCRIPT_DIR,
            env=build_runtime_env(SCRIPT_DIR) if build_runtime_env else None,
        )
        
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        
        if result.returncode != 0:
            print(f"Prediction failed: {result.stderr}")
            return False
        
        print(f"Prediction completed (elapsed: {format_duration(step_duration)})")
        
        # Print prediction-script output (may include sequence split statistics)
        if result.stdout:
            print(result.stdout)
        
        # Extract predicted TF sequences
        tf_fasta = None
        if extract_sequences:
            print("\n2. Extracting predicted TF sequences...")
            step_start_time = time.time()
            
            # Create FASTA output directory
            fasta_output_dir = project_output / "protein_model_preclassification"
            fasta_output_dir.mkdir(exist_ok=True)
            
            prediction_file = output_file
            if not prediction_file.exists():
                print(f"Warning: prediction result file not found: {prediction_file}")
                return False
            
            print(f"Using prediction result file: {prediction_file}")
            tf_fasta = extract_tf_sequences_from_csv(str(processed_fasta), str(prediction_file), str(fasta_output_dir), threshold)
            
            if tf_fasta is None:
                print("Sequence extraction failed")
                return False
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            print(f"TF sequences saved to: {tf_fasta} (elapsed: {format_duration(step_duration)})")
        
        interproscan_success = not run_interproscan_analysis
        hmmscan_success = not run_hmmscan_analysis
        analysis_success = False

        print("\n3. Running protein kinase analysis...")
        step_start_time = time.time()
        pk_success = _run_protein_kinase_analysis(
            processed_fasta,
            project_output,
            enabled=run_protein_kinase_analysis,
        )
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        if pk_success:
            print(f"Protein kinase step completed (elapsed: {format_duration(step_duration)})")
        else:
            return False

        # Run InterProScan
        if run_interproscan_analysis and tf_fasta:
            print("\n4. Running InterProScan...")
            step_start_time = time.time()
            
            interproscan_success = run_interproscan(tf_fasta, str(project_output), appl_list, interproscan_path)
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if interproscan_success:
                print(f"InterProScan completed (elapsed: {format_duration(step_duration)})")
            else:
                print("InterProScan failed")
                return False

        # Run hmmscan
        if run_hmmscan_analysis and tf_fasta:
            print("\n5. Running hmmscan...")
            step_start_time = time.time()
            
            hmmscan_success = run_hmmscan(tf_fasta, str(project_output), interproscan_path)
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if hmmscan_success:
                print(f"hmmscan completed (elapsed: {format_duration(step_duration)})")
            else:
                print("hmmscan failed")
                return False

        # Run classification analysis modules
        if run_interproscan_analysis and run_hmmscan_analysis and tf_fasta:
            print("\n6. Running classification analysis...")
            step_start_time = time.time()
            
            analysis_success = run_analysis_modules(project_output, fasta_file, use_predicted=True, debug=debug, score_threshold=score_threshold, classification_mode=classification_mode)
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if analysis_success:
                print(f"Analysis modules completed (elapsed: {format_duration(step_duration)})")
            else:
                print("Analysis modules failed")
                return False
        
        # 6. Generate Grad-CAM heatmaps
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
                temp_pred_csv = preclass_dir / f"{input_stem}_gradcam_prediction.csv"
                
                grad_cam_cmd = [
                    "python", str(PREDICT_SCRIPT),
                    "--fasta", str(target_fasta),
                    "--threshold", str(threshold),
                    "--output", str(temp_pred_csv),
                    "--project-output", str(project_output),
                    "--mode", predict_mode,
                    "--grad-cam-mode", grad_cam_mode
                ]
                
                if debug:
                    grad_cam_cmd.append("--debug")
                
                print("Executing Grad-CAM generation...")
                gc_result = subprocess.run(grad_cam_cmd, capture_output=True, text=True, cwd=SCRIPT_DIR)
                
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
        project_output = setup_project_output(fasta_file, output)
        
        print("\n=== Starting Sequence Analysis ===\n")
        print(f"Input file: {fasta_file}")
        print(f"Output directory: {project_output}")
        
        processed_fasta = _get_or_create_processed_fasta(fasta_file, project_output)
        interproscan_success = False
        hmmscan_success = False
        analysis_success = False

        print("\n1. Running protein kinase analysis...")
        step_start_time = time.time()

        pk_success = _run_protein_kinase_analysis(
            processed_fasta,
            project_output,
            enabled=run_protein_kinase_analysis,
        )

        step_end_time = time.time()
        step_duration = step_end_time - step_start_time

        if pk_success:
            print(f"Protein kinase step completed (elapsed: {format_duration(step_duration)})")
        else:
            return False

        # Run InterProScan
        print("\n2. Running InterProScan...")
        step_start_time = time.time()
        
        interproscan_success = run_interproscan(processed_fasta, str(project_output), appl_list, interproscan_path)
        
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        
        if interproscan_success:
            print(f"InterProScan completed (elapsed: {format_duration(step_duration)})")
        else:
            print("InterProScan failed")
            return False

        # Run hmmscan
        print("\n3. Running hmmscan...")
        step_start_time = time.time()
        
        hmmscan_success = run_hmmscan(processed_fasta, str(project_output), interproscan_path)
        
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        
        if hmmscan_success:
            print(f"hmmscan completed (elapsed: {format_duration(step_duration)})")
        else:
            print("hmmscan failed")
            return False

        # Run classification
        if interproscan_success and hmmscan_success:
            print("\n4. Running classification analysis...")
            step_start_time = time.time()
            
            analysis_success = run_analysis_modules(project_output, fasta_file, use_predicted=False, debug=debug, score_threshold=score_threshold, classification_mode=classification_mode)
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            if analysis_success:
                print(f"Analysis modules completed (elapsed: {format_duration(step_duration)})")
            else:
                print("Analysis modules failed")
                return False
        else:
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
        project_output = setup_project_output(fasta_file, output)
        print(f"Project output directory: {project_output}")
        
        # Create required subdirectories
        ipr_dir = project_output / "InterproScan"
        hmmscan_dir = project_output / "hmmscan"
        result_dir = project_output / "result"
        
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
        analysis_success = run_analysis_modules(project_output, fasta_file, use_predicted=False, debug=debug, score_threshold=score_threshold, classification_mode=classification_mode)
        
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
