import os
import json
import argparse
import gzip
from Bio import SeqIO
from Bio.Seq import Seq

NUCLEOTIDE_ALPHABET = set("ACGTUNWSMKRYBDHV")
PROTEIN_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYXBZJUO")
START_CODONS = {"ATG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}


def normalize_sequence(seq):
    return "".join(str(seq).split()).upper()


def strip_compression_suffix(path):
    path = str(path)
    if path.endswith(".gz"):
        return path[:-3]
    return path


def get_input_stem(path):
    return os.path.splitext(os.path.basename(strip_compression_suffix(path)))[0]


def open_fasta_text(path, mode="rt", encoding="utf-8"):
    path = str(path)
    if "b" not in mode and encoding is None:
        encoding = "utf-8"
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=encoding)
    return open(path, mode, encoding=encoding)


def sanitize_protein_sequence(seq):
    seq = normalize_sequence(seq)
    if not seq:
        return seq
    seq = seq.rstrip("*")
    return seq.replace("*", "X")


def _has_terminal_stop(seq):
    seq = normalize_sequence(seq).replace("U", "T")
    if len(seq) < 3:
        return False
    last_full_codon = seq[(len(seq) // 3 - 1) * 3:(len(seq) // 3) * 3]
    return last_full_codon in STOP_CODONS


def _is_likely_nucleotide_sequence(seq):
    seq = normalize_sequence(seq).replace("U", "T")
    if not seq:
        return False
    if not set([c for c in seq if c.isalpha()]).issubset(NUCLEOTIDE_ALPHABET):
        return False
    if seq.startswith("ATG"):
        return True
    if len(seq) % 3 == 0 and _has_terminal_stop(seq):
        return True
    return False


def classify_input_sequence(seq):
    seq = normalize_sequence(seq)
    letters = [c for c in seq if c.isalpha()]
    if not letters:
        return "empty"

    charset = set(letters)
    nucleotide_like = charset.issubset(NUCLEOTIDE_ALPHABET)
    protein_like = charset.issubset(PROTEIN_ALPHABET)

    if nucleotide_like and _is_likely_nucleotide_sequence(seq):
        return "nucleotide"
    if protein_like:
        return "protein"
    if nucleotide_like:
        return "nucleotide"
    return "invalid"


def validate_protein_sequence(seq, record_id=None):
    seq = sanitize_protein_sequence(seq)
    if not seq:
        raise ValueError("is empty")
    invalid_non_letters = sorted({c for c in seq if not c.isalpha()})
    if invalid_non_letters:
        raise ValueError(f"contains unsupported characters: {', '.join(invalid_non_letters)}")

    invalid_chars = sorted({c for c in seq if c.isalpha() and c not in PROTEIN_ALPHABET})
    if invalid_chars:
        raise ValueError(f"contains non-protein characters: {', '.join(invalid_chars)}")
    return seq

def get_output_dir(base_dir=None):
    """Get the output directory path.
    
    Args:
        base_dir (str, optional): Base directory; if None, use the project root.
    
    Returns:
        str: Absolute path to the result directory.
    """
    if base_dir is None:
        # Use the parent directory of this module as the project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
    
    # Build result directory path
    output_dir = os.path.join(base_dir, "result")
    
    # Create it if missing
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir

def get_output_subdir(name, base_dir=None):
    base = get_output_dir(base_dir)
    sub = os.path.join(base, name)
    if not os.path.exists(sub):
        os.makedirs(sub)
    return sub

def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_project_output_dir(fasta_file):
    root = _repo_root()
    output_base = os.path.join(root, "output")
    base = get_input_stem(fasta_file)
    if not os.path.exists(output_base):
        os.makedirs(output_base)
    candidates = []
    for name in os.listdir(output_base):
        if name == base or (name.startswith(base + "_")):
            full = os.path.join(output_base, name)
            if os.path.isdir(full):
                candidates.append(full)
    if candidates:
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]
    target = os.path.join(output_base, base)
    os.makedirs(target, exist_ok=True)
    return target

def read_fasta_to_dict(fasta_file):
    """
    Read a FASTA file and convert it to a dictionary.
    
    Args:
        fasta_file (str): Path to a FASTA file.
    
    Returns:
        dict: Mapping from record ID to sequence string.
    """
    fasta_dict = {}
    
    try:
        with open_fasta_text(fasta_file, 'rt') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                # Use record.id as the key and the sequence as the value
                fasta_dict[record.id] = str(record.seq)
        
        print(f"Successfully read {len(fasta_dict)} sequences")
        return fasta_dict
        
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        return {}

def format_tf_fasta_with_classification(fasta_file, classification_result, output_file):
    """
    Generate a standardized FASTA file with TF classification annotations.
    
    Args:
        fasta_file (str): Input FASTA file path.
        classification_result (dict): Classification results, keyed by gene_id.
        output_file (str): Output FASTA file path.
    
    Returns:
        bool: True on success, otherwise False.
    """
    try:
        fasta_dict = read_fasta_to_dict(fasta_file)
        if not fasta_dict:
            print("Unable to read the input FASTA file for classification")
            return False
        
        # Write only sequences that have classification results
        with open(output_file, 'w') as f:
            written_count = 0
            for gene_id, sequence in fasta_dict.items():
                # Only include entries present in the classification map
                if gene_id in classification_result:
                    tf_info = classification_result[gene_id]
                    
                    # FASTA header format: >gene_id | family | type
                    header = f">{gene_id} | {tf_info['family']} | {tf_info['type']}"
                    
                    # Write header and sequence
                    f.write(header + '\n')
                    f.write(sequence + '\n')
                    written_count += 1
            
            print(f"Successfully generated a FASTA file containing {written_count} classified sequences")
            print("All output sequences include TF classification annotations")
            return True
            
    except Exception as e:
        print(f"Error generating FASTA file: {e}")
        return False

def generate_classified_fasta(fasta_file, classification_result, output_file=None, output_dir=None):
    """
    Generate a FASTA file with TF classification annotations (for use by other modules).

    Only sequences with classification results are included.
    
    Args:
        fasta_file (str): Input FASTA file path.
        classification_result (dict): In-memory classification results.
        output_file (str, optional): Output FASTA file path. If None, a default is generated.
        output_dir (str, optional): Output directory path (e.g., project subdirectory).
    
    Returns:
        str: Output file path; returns None on failure.
    """
    try:
        # If output_file is not provided, generate a default path
        if not output_file:
            input_name = get_input_stem(fasta_file)
            if output_dir:
                result_dir = output_dir
            else:
                result_dir = get_output_dir(output_dir)
            output_file = os.path.join(result_dir, f"{input_name}_tf_classified.fasta")
        
        # Run formatter
        success = format_tf_fasta_with_classification(fasta_file, classification_result, output_file)
        
        if success:
            return output_file
        else:
            return None
            
    except Exception as e:
        print(f"Error generating classified FASTA file: {e}")
        return None

def _looks_like_cds(seq):
    return classify_input_sequence(seq) == "nucleotide"

def _best_six_frame_translate(seq, table=1, min_orf_aa=30):
    s = Seq(seq)
    candidates = []
    for frame in (0, 1, 2):
        aa = str(s[frame:].translate(table=table, to_stop=False))
        seg = max(aa.split('*'), key=len) if aa else ""
        candidates.append((seg, f"+{frame+1}"))
    rc = s.reverse_complement()
    for frame in (0, 1, 2):
        aa = str(rc[frame:].translate(table=table, to_stop=False))
        seg = max(aa.split('*'), key=len) if aa else ""
        candidates.append((seg, f"-{frame+1}"))
    candidates.sort(key=lambda x: len(x[0]), reverse=True)
    best = candidates[0]
    if len(best[0]) < min_orf_aa:
        return "", best[1]
    return best

def _extract_orfs_from_frame(frame_seq, frame_label, table=1, min_orf_aa=30):
    codons = [frame_seq[i:i + 3] for i in range(0, len(frame_seq) - 2, 3)]
    if not codons:
        return []

    aa_seq = str(Seq("".join(codons)).translate(table=table, to_stop=False))
    results = []
    current_start = None
    orf_index = 0

    for idx, (codon, aa_char) in enumerate(zip(codons, aa_seq)):
        if current_start is None and codon in START_CODONS:
            current_start = idx

        if aa_char == "*":
            if current_start is not None:
                peptide = aa_seq[current_start:idx]
                if len(peptide) >= min_orf_aa:
                    orf_index += 1
                    results.append((peptide, f"{frame_label}|orf={orf_index}"))
            current_start = None

    return results


def _extract_translated_orfs(seq, table=1, min_orf_aa=30):
    s = Seq(normalize_sequence(seq).replace("U", "T"))
    results = []

    for frame in (0, 1, 2):
        frame_seq = str(s[frame:])
        results.extend(_extract_orfs_from_frame(frame_seq, f"+{frame+1}", table=table, min_orf_aa=min_orf_aa))

    rc = s.reverse_complement()
    for frame in (0, 1, 2):
        frame_seq = str(rc[frame:])
        results.extend(_extract_orfs_from_frame(frame_seq, f"-{frame+1}", table=table, min_orf_aa=min_orf_aa))

    return results

def generate_protein_sequences_in_memory(fasta_file, genetic_code=1, min_orf_aa=30):
    try:
        sequences = []
        with open_fasta_text(fasta_file, 'rt') as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = normalize_sequence(record.seq)
                seq_type = classify_input_sequence(seq)
                if seq_type == "nucleotide":
                    translated = _extract_translated_orfs(seq, table=genetic_code, min_orf_aa=min_orf_aa)
                    for aa, frame in translated:
                        sequences.append({"header": f"{record.id}|{frame}", "sequence": aa})
                elif seq_type == "protein":
                    sequences.append({"header": record.id, "sequence": validate_protein_sequence(seq, record.id)})
                else:
                    raise ValueError(f"Sequence {record.id} is neither a valid protein nor a likely CDS input")
        return sequences
    except Exception as e:
        print(f"Failed to generate in-memory protein sequences: {e}")
        return []

def format_tf_fasta_with_classification_from_mem(seqs_dict, classification_result, output_file):
    try:
        with open(output_file, 'w') as f:
            written_count = 0
            for gene_id, sequence in seqs_dict.items():
                if gene_id in classification_result:
                    tf_info = classification_result[gene_id]
                    header = f">{gene_id} | {tf_info['family']} | {tf_info['type']}"
                    f.write(header + '\n')
                    f.write(sequence + '\n')
                    written_count += 1
        print(f"Successfully generated a FASTA file containing {written_count} classified sequences")
        print("All output sequences include TF classification annotations")
        return True
    except Exception as e:
        print(f"Error generating FASTA file (in-memory): {e}")
        return False
def get_processed_fasta_path(fasta_file, output_dir=None):
    input_name = get_input_stem(fasta_file)
    project_dir = get_project_output_dir(fasta_file) if output_dir is None else output_dir
    # Previously: write into a "six_frame_translation" subdirectory.
    # Now: write directly under the project output directory.
    return os.path.join(project_dir, f"{input_name}_protein_replaced.fasta")

def generate_protein_fasta_with_translation(fasta_file, output_file=None, output_dir=None, genetic_code=1, min_orf_aa=30):
    try:
        if not output_file:
            output_file = get_processed_fasta_path(fasta_file, output_dir)
        total = 0
        kept_proteins = 0
        translated_orfs = 0
        nucleotide_entries = 0
        skipped_nucleotide_entries = 0

        with open_fasta_text(fasta_file, 'rt') as handle, open(output_file, 'w') as out:
            for record in SeqIO.parse(handle, "fasta"):
                total += 1
                seq = normalize_sequence(record.seq)
                seq_type = classify_input_sequence(seq)

                if seq_type == "nucleotide":
                    nucleotide_entries += 1
                    translated = _extract_translated_orfs(seq, table=genetic_code, min_orf_aa=min_orf_aa)
                    if not translated:
                        skipped_nucleotide_entries += 1
                        print(
                            f"[WARN] No complete ORF meeting min_orf_aa={min_orf_aa} was found for nucleotide sequence {record.id}; skipping"
                        )
                        continue
                    for aa, frame in translated:
                        out.write(f">{record.id}|{frame}\n")
                        out.write(aa + "\n")
                        translated_orfs += 1
                elif seq_type == "protein":
                    protein_seq = validate_protein_sequence(seq, record.id)
                    out.write(f">{record.id}\n")
                    out.write(protein_seq + "\n")
                    kept_proteins += 1
                else:
                    raise ValueError(f"Sequence {record.id} is neither a valid protein nor a likely CDS input")

        print(
            f"Read {total} sequences; retained {kept_proteins} protein sequences; generated {translated_orfs} ORF-derived protein sequences from {nucleotide_entries} nucleotide entries"
        )
        if skipped_nucleotide_entries:
            print(f"Skipped {skipped_nucleotide_entries} nucleotide entries without complete ORFs")
        print(f"Written to: {output_file}")
        return output_file
    except Exception as e:
        if output_file and os.path.exists(output_file):
            try:
                os.remove(output_file)
            except OSError:
                pass
        print(f"Failed to write processed protein FASTA: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate a FASTA file with TF annotations or preprocess mixed protein/CDS input into protein FASTA')
    parser.add_argument('-i', '--input', required=True, help='Path to input FASTA file')
    parser.add_argument('-o', '--output', help='Path to output FASTA file (optional)')
    parser.add_argument('--classification', help='Path to classification-result JSON file (for testing)')
    parser.add_argument('--translate-only', action='store_true', help='Preprocess mixed protein/CDS input and write protein FASTA output')
    parser.add_argument('--genetic-code', type=int, default=1, help='Genetic code table ID used for translation')
    parser.add_argument('--min-orf-aa', type=int, default=30, help='Minimum complete ORF length threshold (amino acids)')
    
    args = parser.parse_args()
    
    input_name = os.path.splitext(os.path.basename(args.input))[0]
    if not args.output:
        if args.translate_only:
            args.output = get_processed_fasta_path(args.input)
        else:
            output_dir = get_output_dir()
            args.output = os.path.join(output_dir, f"{input_name}_tf_classified.fasta")

    if args.translate_only:
        out = generate_protein_fasta_with_translation(
            args.input,
            output_file=args.output,
            output_dir=None,
            genetic_code=args.genetic_code,
            min_orf_aa=args.min_orf_aa,
        )
        if out:
            print(f"Protein FASTA saved to: {out}")
        else:
            print("Input preprocessing failed")
    else:
        if args.classification:
            try:
                with open(args.classification, 'r', encoding='utf-8') as f:
                    classification_result = json.load(f)
                print(f"Loaded {len(classification_result)} classification results from file")
            except Exception as e:
                print(f"Error reading classification-result file: {e}")
                classification_result = {}
        else:
            classification_result = {}
            print("No classification results provided; generating a FASTA file without TF annotations")
        success = format_tf_fasta_with_classification(args.input, classification_result, args.output)
        if success:
            print(f"FASTA file saved to: {args.output}")
        else:
            print("FASTA generation failed")

if __name__ == "__main__":
    main()
