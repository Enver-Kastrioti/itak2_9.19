import json
import os
import argparse
from collections import defaultdict

def parse_pfam_spec(file_path):
    # Use defaultdict to store the parsed results
    result = defaultdict(lambda: defaultdict(list))
    hit_counts = defaultdict(lambda: defaultdict(int))
    temp_scores = defaultdict(dict)  # Temporary storage for NF-YB/NF-YC scores
    
    # Read the file and process it line by line
    with open(file_path, 'r') as f:
        for line in f:
            # Skip comment lines and empty lines
            if line.startswith('#') or not line.strip():
                continue
            
            # Split the line into fields
            fields = line.strip().split()
            if len(fields) < 9:
                continue
                
            # Extract required fields
            accession = fields[1]  # Column 2
            gene_id = fields[2]    # Column 3
            evalue = fields[7]     # Column 8
            score = fields[8]      # Column 9
            
            # Enforce a minimum score threshold
            try:
                score_value = float(score)
                if score_value < 20:
                    continue
            except ValueError:
                continue

            # Track hit counts
            hit_counts[gene_id][accession] += 1
            
            # Build a hit record consistent with the IPR-derived schema
            hit = {
                "accession": accession,
                "library": "selfbuild",
                "ipr": "",  # hmmscan output does not provide IPR identifiers
                "ipr_name": "",  # hmmscan output does not provide IPR names
                "description": accession,  # Use accession as the description
                "start": "",  # hmmscan output does not provide coordinates here
                "end": "",    # hmmscan output does not provide coordinates here
                "evalue": evalue,
                "score": score
            }
            
            # For repeated hits, annotate keys using the "&<count>" suffix
            count = hit_counts[gene_id][accession]
            if count > 1:
                # Add a count-suffixed entry
                key = f"{accession}&{count}"
                result[gene_id][key].append(hit)
                
                # Back-fill empty placeholders for earlier counts
                for i in range(count-1, 0, -1):
                    empty_key = f"{accession}&{i}" if i > 1 else accession
                    if not result[gene_id][empty_key]:
                        result[gene_id][empty_key] = []
            else:
                result[gene_id][accession].append(hit)
    
    # Convert to a structure consistent with cl_json.json
    match_list = []
    for gene_id, gene_data in result.items():
        # The entry is a two-element list:
        # (1) an empty dict placeholder for sequence metadata
        # (2) a domain-to-matches dictionary
        gene_entry = {
            gene_id: [
                {},  # Placeholder for sequence metadata
                dict(gene_data)  # Domain-to-matches dictionary
            ]
        }
        match_list.append(gene_entry)
    
    formatted_result = {
        "result": {
            "match": match_list
        }
    }
    
    return formatted_result

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Process PFAM specificity analysis results')
    parser.add_argument('-i', '--input', required=True, help='Path to the input file (hmmscan result file)')
    parser.add_argument('-o', '--output', required=True, help='Path to the output directory')
    
    args = parser.parse_args()
    
    input_file = args.input
    output_dir = args.output
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_file = os.path.join(output_dir, "pfamspec.json")
    
    try:
        # Parse the input file
        result = parse_pfam_spec(input_file)
        
        # Write results to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        raise

if __name__ == "__main__":
    main()
