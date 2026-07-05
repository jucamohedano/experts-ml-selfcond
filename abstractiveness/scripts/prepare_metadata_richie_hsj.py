import json
import pathlib

import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WORDLIST_CSV = REPO_ROOT / "assets" / "Richie_and_Bhatia-HSJ" / "table 1 - word lists.csv"
WIKI_FILE = REPO_ROOT / "assets" / "enwiki-2023-04-13.txt"
TYPICALITY_FILE = REPO_ROOT / "assets" / "THINGS-database" / "osfstorage" / "03_category-level" / "typicality53_mean-ratings.tsv"
OUTPUT_JSON = REPO_ROOT / "assets" / "metadata_Richie_HSJ.json"

# Maps this word-list's category names to the corresponding THINGS
# superordinate category name. A typicality score is only used when both
# the word AND its THINGS category match this mapping (no cross-category
# averaging). Categories with no THINGS equivalent (sports, professions)
# always resolve to null.
RICHIE_TO_THINGS_CATEGORY = {
    "furniture": "furniture",
    "clothing": "clothing",
    "birds": "bird",
    "vegetables": "vegetable",
    "sports": None,
    "vehicles": "vehicle",
    "fruit": "fruit",
    "professions": None,
}


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


def load_categories_from_wordlist(csv_path):
    """Load category -> [members] from the Richie & Bhatia HSJ table 1 word-list CSV."""
    df = pd.read_csv(csv_path)
    categories = {}
    for col in df.columns:
        members = df[col].dropna().astype(str).str.strip().str.lower().tolist()
        members = [m for m in members if m]
        categories[col.strip().lower()] = members
    return categories


def prepare_metadata(wordlist_path, freq_path, typ_path, out_json_path):
    """Generates the metadata list from the Richie & Bhatia HSJ word lists, saving to JSON."""

    # Load frequencies
    wiki_freq = load_frequencies_from_file(freq_path)

    # Load typicality scores, keyed by (member, THINGS category) so that a
    # word rated under multiple unrelated THINGS categories (e.g. "cabinet"
    # under both "container" and "furniture") is not blended into a single
    # cross-category average.
    typ_dict = {}
    if pathlib.Path(typ_path).exists():
        df_typ = pd.read_csv(typ_path, sep="\t")
        df_typ['member_lower'] = df_typ['member'].str.lower()
        df_typ['category_lower'] = df_typ['category'].str.lower()
        typ_dict = df_typ.set_index(['member_lower', 'category_lower'])['typicality_score'].to_dict()
    else:
        print(f"Warning: Typicality file {typ_path} not found.")

    # Load word-list data
    categories = load_categories_from_wordlist(wordlist_path)
    metadata_list = []

    # Build metadata list
    for cat_name, concepts in categories.items():
        cat_name_lower = cat_name.lower()

        # --- LEVEL 1: ROOT CATEGORIES ---
        metadata_list.append({
            "concept": cat_name_lower,
            "abstraction_level": 1,
            "frequency": wiki_freq.get(cat_name_lower, None),
            "typicality": None,  # Intentionally None for root categories
            "category": None     # Parent category is None for root categories
        })

        # --- LEVEL 2: SPECIFIC CONCEPTS ---
        things_category = RICHIE_TO_THINGS_CATEGORY.get(cat_name_lower)
        for concept in concepts:
            concept_lower = concept.lower()

            # Fetch typicality only when the word is rated under the THINGS
            # category matching this word-list's category; round to 3
            # decimals for cleanliness (if it exists)
            t_score = None
            if things_category is not None:
                t_score = typ_dict.get((concept_lower, things_category), None)
            if t_score is not None:
                t_score = round(t_score, 3)

            metadata_list.append({
                "concept": concept_lower,
                "abstraction_level": 2,
                "frequency": wiki_freq.get(concept_lower, None),
                "typicality": t_score,
                "category": cat_name_lower  # Set to the parent category
            })

    # Write to JSON file
    with open(out_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_list, f, indent=4)

    print(f"Success! Generated {len(metadata_list)} metadata entries and saved to {out_json_path}.")
    return metadata_list


if __name__ == "__main__":
    generated_metadata = prepare_metadata(WORDLIST_CSV, WIKI_FILE, TYPICALITY_FILE, OUTPUT_JSON)

    # Print a quick preview of the first few items to verify
    print("\nPreview of the first 5 entries:")
    print(json.dumps(generated_metadata[:5], indent=4))
