#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iTAK3 FASTA validation module.

Validates whether the input FASTA file is well-formed, contains protein sequences,
and does not include disallowed characters (e.g., '*').
"""

import os
import re
from pathlib import Path
from Bio import SeqIO

try:
    from module.get_fasta import (
        NUCLEOTIDE_ALPHABET,
        PROTEIN_ALPHABET,
        classify_input_sequence,
        open_fasta_text,
        normalize_sequence,
        validate_protein_sequence,
    )
except ImportError:
    from get_fasta import (
        NUCLEOTIDE_ALPHABET,
        PROTEIN_ALPHABET,
        classify_input_sequence,
        open_fasta_text,
        normalize_sequence,
        validate_protein_sequence,
    )

class FastaValidator:
    """FASTA validator."""
    
    def __init__(self):
        self.errors = []
        self.protein_count = 0
        self.nucleotide_count = 0
        
        # Canonical amino-acid alphabet
        self.valid_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
        # Extended amino-acid alphabet (including ambiguity codes)
        self.extended_amino_acids = set(PROTEIN_ALPHABET)
        self.valid_nucleotides = set(NUCLEOTIDE_ALPHABET)
        

    
    def validate_fasta_format(self, fasta_file):
        try:
            with open_fasta_text(fasta_file, "rt") as handle:
                sequences = list(SeqIO.parse(handle, "fasta"))
            if not sequences:
                self.errors.append("The file contains no FASTA records")
                return False
            
            sequence_count = len(sequences)
            print(f"  [#] Found {sequence_count} sequences")
            return True
            
        except Exception as e:
            self.errors.append(f"FASTA parsing error: {str(e)}")
            return False
    
    def validate_protein_sequences(self, fasta_file):
        try:
            with open_fasta_text(fasta_file, "rt") as handle:
                sequences = list(SeqIO.parse(handle, "fasta"))
            self.protein_count = 0
            self.nucleotide_count = 0
            
            for record in sequences:
                seq_str = normalize_sequence(record.seq)

                seq_type = classify_input_sequence(seq_str)
                if seq_type == "empty":
                    self.errors.append(f"Sequence {record.id} is empty")
                    continue
                if seq_type != "protein":
                    self.errors.append(f"Sequence {record.id} looks like nucleotide input; this FASTA must contain proteins only")
                    continue
                try:
                    validate_protein_sequence(seq_str, record.id)
                except ValueError as exc:
                    self.errors.append(f"Sequence {record.id} {exc}")
                    continue

                self.protein_count += 1

            if self.protein_count == 0:
                self.errors.append("No valid protein sequences were found")
                return False

            print(f"  [#] Found {self.protein_count} valid protein sequences")
            return True
        except Exception as e:
            self.errors.append(f"Error during sequence validation: {str(e)}")
            return False

    def validate_input_sequences(self, fasta_file):
        try:
            with open_fasta_text(fasta_file, "rt") as handle:
                sequences = list(SeqIO.parse(handle, "fasta"))
            self.protein_count = 0
            self.nucleotide_count = 0

            for record in sequences:
                seq_str = normalize_sequence(record.seq)

                seq_type = classify_input_sequence(seq_str)
                if seq_type == "protein":
                    try:
                        validate_protein_sequence(seq_str, record.id)
                    except ValueError as exc:
                        self.errors.append(f"Sequence {record.id} {exc}")
                        continue
                    self.protein_count += 1
                elif seq_type == "nucleotide":
                    self.nucleotide_count += 1
                elif seq_type == "empty":
                    self.errors.append(f"Sequence {record.id} is empty")
                else:
                    letters = sorted({c for c in seq_str if c.isalpha()})
                    invalid_chars = sorted(set(letters) - self.extended_amino_acids - self.valid_nucleotides)
                    invalid_non_letters = sorted({c for c in seq_str if not c.isalpha()})
                    if invalid_chars:
                        self.errors.append(
                            f"Sequence {record.id} contains unsupported characters: {', '.join(invalid_chars)}"
                        )
                    elif invalid_non_letters:
                        self.errors.append(
                            f"Sequence {record.id} contains unsupported characters: {', '.join(invalid_non_letters)}"
                        )
                    else:
                        self.errors.append(
                            f"Sequence {record.id} could not be classified as either protein or CDS input"
                        )

            if self.protein_count + self.nucleotide_count == 0:
                self.errors.append("No valid protein or nucleotide sequences were found")
                return False

            print(
                f"  [#] Found {self.protein_count} protein sequences and {self.nucleotide_count} nucleotide sequences"
            )
            return True
        except Exception as e:
            self.errors.append(f"Error during sequence validation: {str(e)}")
            return False

    def validate_sequences(self, sequences):
        try:
            self.protein_count = 0
            self.nucleotide_count = 0
            for item in sequences:
                seq_str = normalize_sequence(item if isinstance(item, str) else item.get('sequence', ''))
                if not seq_str:
                    self.errors.append("Empty sequence")
                    continue
                seq_type = classify_input_sequence(seq_str)
                if seq_type != "protein":
                    self.errors.append("Sequence is not a valid protein sequence")
                    continue
                try:
                    validate_protein_sequence(seq_str)
                except ValueError as exc:
                    self.errors.append(f"Sequence {exc}")
                    continue
                self.protein_count += 1
            if self.protein_count == 0:
                self.errors.append("No valid protein sequences were found")
                return False
            return True
        except Exception as e:
            self.errors.append(f"Error during sequence validation: {str(e)}")
            return False
    

    
    def run_full_validation(self, fasta_file, allow_nucleotide=True):

        print(f"[#] Starting FASTA validation: {fasta_file}")
        
        # Reset error state
        self.errors = []
        
        # 1) Basic FASTA format validation
        if not self.validate_fasta_format(fasta_file):
            print("[WARN] FASTA format validation failed")
            return False
        
        # 2) Sequence-content validation
        if allow_nucleotide:
            content_valid = self.validate_input_sequences(fasta_file)
            validation_label = "Input sequence"
        else:
            content_valid = self.validate_protein_sequences(fasta_file)
            validation_label = "Protein sequence"

        if not content_valid:
            print(f"[WARN] {validation_label} validation failed")
            return False
        
        print("[#] FASTA validation passed")
        return True
    


def main():
    """Entry point."""
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python validate_fasta.py <fasta_file>")
        sys.exit(1)
    
    fasta_file = sys.argv[1]
    validator = FastaValidator()
    success = validator.run_full_validation(fasta_file)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
