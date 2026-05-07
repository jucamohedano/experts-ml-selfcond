import yaml
import json
import pandas as pd
import pathlib

# --- Configuration ---
YAML_FILE = "../conf/concept_group/abstractiveness_150.yaml"
WIKI_FILE = "../assets/enwiki-2023-04-13.txt"
TYPICALITY_FILE = "../assets/THINGS-database/osfstorage/03_category-level/typicality53_mean-ratings.tsv"
OUTPUT_JSON = "../assets/metadata.json"

def load_frequencies_from_file(file_path):
    """Load word frequencies from Wikipedia count file."""
    freq_dict = {}
    p = pathlib.Path(file_path)
    if not p.exists():
        print(f"Warning: {file_path} not found.")
        return freq_dict

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            parts = line.rsplit(None, 1) 
            if len(parts) == 2:
                word, count = parts
                try:
                    freq_dict[word.lower()] = int(count)
                except ValueError: 
                    continue
    return freq_dict

def prepare_metadata(yaml_path, freq_path, typ_path, out_json_path):
    """Generates the metadata list from YAML and datasets, saving to JSON."""
    
    # Load frequencies
    wiki_freq = load_frequencies_from_file(freq_path)
    
    # Load typicality scores
    typ_dict = {}
    if pathlib.Path(typ_path).exists():
        df_typ = pd.read_csv(typ_path, sep="\t")
        df_typ['member_lower'] = df_typ['member'].str.lower()
        # Group by member and take the mean typicality just in case of duplicates
        typ_dict = df_typ.groupby('member_lower')['typicality_score'].mean().to_dict()
    else:
        print(f"Warning: Typicality file {typ_path} not found.")

    # Load YAML data
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.safe_load(f)
        
    categories = yaml_data.get('categories', {})
    metadata_list = []

    # Build metadata list
    for cat_name, concepts in categories.items():
        cat_name_lower = cat_name.lower()
        
        # --- LEVEL 1: ROOT CATEGORIES ---
        metadata_list.append({
            "concept": cat_name_lower,
            "abstraction_level": 1,
            "frequency": wiki_freq.get(cat_name_lower, None),
            "typicality": None, # Intentionally None for root categories
            "category": None    # Parent category is None for root categories
        })
        
        # --- LEVEL 2: SPECIFIC CONCEPTS ---
        for concept in concepts:
            concept_lower = concept.lower()
            
            # Fetch typicality and round to 3 decimals for cleanliness (if it exists)
            t_score = typ_dict.get(concept_lower, None)
            if t_score is not None:
                t_score = round(t_score, 3)
                
            metadata_list.append({
                "concept": concept_lower,
                "abstraction_level": 2,
                "frequency": wiki_freq.get(concept_lower, None),
                "typicality": t_score,
                "category": cat_name_lower # Set to the parent category
            })
            
    # Write to JSON file
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4)
        
    print(f"Success! Generated {len(metadata_list)} metadata entries and saved to {out_json_path}.")
    return metadata_list

if __name__ == "__main__":
    generated_metadata = prepare_metadata(YAML_FILE, WIKI_FILE, TYPICALITY_FILE, OUTPUT_JSON)
    
    # Print a quick preview of the first few items to verify
    print("\nPreview of the first 5 entries:")
    print(json.dumps(generated_metadata[:5], indent=4))