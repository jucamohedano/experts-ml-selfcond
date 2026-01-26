import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch.nn.functional as F
import numpy as np
import pickle
import logging
import pathlib
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.corpus import wordnet as wn
import spacy
import re
from typing import List, Optional

# Add project root to path
sys.path.append(os.getcwd())

log = logging.getLogger(__name__)

NEUTRAL_PROMPTS = [
    "Once upon a time",
    "The story begins with",
    "In the beginning,",
    "Long ago,",
    "Yesterday,",
]

class SteeringHook:
    def __init__(self, vector, coeff, max_act=1.0):
        self.vector = vector
        self.coeff = coeff
        self.max_act = max_act
        
    def __call__(self, module, input, output):
        # output is usually a tuple (hidden_states, ...)
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
            
        # Inject vector: h' = h + coeff * v
        # Vector shape: [units]
        # Hidden states shape: [batch, seq_len, units]
        
        # Ensure vector is on the same device and type
        v = self.vector.to(hidden_states.device).to(hidden_states.dtype)
        
        # Add to all tokens
        # Scale by max_act if provided (and coeff)
        # Effective Vector = Factor * Max_Activation * Normalized_Vector
        hidden_states += self.coeff * self.max_act * v
        
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

class LexicalFieldScorer:
    def __init__(self):
        log.info("Initializing LexicalFieldScorer...")
        try:
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            log.info("Downloading WordNet...")
            nltk.download('wordnet')
            
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            log.info("Downloading spacy model en_core_web_sm...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def get_lexical_field(self, concept: str, max_words: int = 50) -> List[str]:
        """
        Get related words from WordNet, prioritizing concrete object senses 
        (Artifacts, Living Things, Body Parts) to match dataset condition.
        """
        # Define target categories (Artifacts, Living Things, Body Parts)
        # We ensure these can be loaded (after __init__ download check)
        try:
            target_hypernyms = {
                wn.synset('artifact.n.01'), 
                wn.synset('living_thing.n.01'),
                wn.synset('body_part.n.01')
            }
        except Exception as e:
            log.warning(f"Error loading target hypernyms: {e}. using all synsets.")
            target_hypernyms = set()

        def is_relevant(synset):
            if not target_hypernyms: return True
            for hypernym in synset.closure(lambda s: s.hypernyms()):
                if hypernym in target_hypernyms:
                    return True
            return False

        related = set()
        related.add(concept.lower())
        
        # Get noun synsets
        synsets = wn.synsets(concept, pos=wn.NOUN)
        
        # Select relevant synsets
        relevant_synsets = [s for s in synsets if is_relevant(s)]
        
        if relevant_synsets:
            # Prioritize the most frequent (first) relevant sense
            target_synsets = [relevant_synsets[0]]
        elif synsets:
            # Fallback to first sense if no relevant concrete sense found
            target_synsets = [synsets[0]]
        else:
            # No synsets found at all
            return list(related)

        for synset in target_synsets:
            # Add synonyms
            related.update(lemma.name().replace('_', ' ').lower() 
                          for lemma in synset.lemmas())
            # Add hyponyms (more specific terms)
            for hypo in synset.hyponyms():
                related.update(lemma.name().replace('_', ' ').lower() 
                             for lemma in hypo.lemmas())
        
        return list(related)[:max_words]

    def extract_content_words(self, text: str) -> List[str]:
        """Extract content words (nouns, verbs, adjectives, adverbs)."""
        doc = self.nlp(text.lower())
        content_words = [
            token.lemma_ 
            for token in doc 
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']
        ]
        return content_words

    def score(self, text: str, concept: str) -> float:
        """
        Compute percentage of content words in text that match the concept's lexical field.
        """
        content_words = self.extract_content_words(text)
        if not content_words:
            return 0.0
            
        lexical_field = self.get_lexical_field(concept)
        # Lemmatize lexical field for better matching (although wn lemmas are usually base form)
        field_lemmas = set([self.nlp(w)[0].lemma_ for w in lexical_field])
        
        # Count matches
        matches = sum(1 for w in content_words if w in field_lemmas)
        
        return matches / len(content_words)

def extract_layer_from_vector_path(vector_path):
    """
    Extracts the layer name from a vector file path.
    Example: vectors_transformer.h.6.attn.c_proj_global.pkl -> transformer.h.6.attn.c_proj
    """
    path = pathlib.Path(vector_path)
    stem = path.stem  # Remove .pkl extension
    
    # Remove "vectors_" prefix
    if stem.startswith("vectors_"):
        stem = stem[len("vectors_"):]
    
    # Remove known suffixes
    for suffix in ["_global", "_masked", "_killed"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
            
    return stem

def load_vector(vector_path, concept):
    """
    Loads a concept vector from a pickle file.
    Expects the pickle to contain a dictionary {concept: vector} or {concept: {'vector': ..., 'max_act': ...}}.
    Returns (vector, max_act).
    """
    path = pathlib.Path(vector_path)
    if not path.exists():
        raise FileNotFoundError(f"Vector file not found: {path}")
        
    with open(path, "rb") as f:
        data = pickle.load(f)
        
    if isinstance(data, dict):
        if concept in data:
            item = data[concept]
            if isinstance(item, dict) and 'vector' in item:
                return item['vector'], item.get('max_act', 1.0)
            else:
                return item, 1.0 # Legacy format or just vector
        else:
            raise KeyError(f"Concept '{concept}' not found. Available: {list(data.keys())}")
    elif isinstance(data, np.ndarray):
        return data, 1.0
    else:
        raise ValueError(f"Unknown vector format: {type(data)}")

def get_latest_timestamp_dir(base_path: pathlib.Path) -> Optional[pathlib.Path]:
    """Find the subdirectory with the latest timestamp."""
    if not base_path.exists():
        return None
    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    # Assuming standard YYYY-MM-DD_HH-MM-SS format, lexical sort works for time
    return max(subdirs, key=lambda p: p.name)

def run_steering_validation(vector_base_dir, concept, concept_types, layer_indices, 
                            layer_types, vector_suffixes, coeff_start, coeff_end, 
                            coeff_steps, max_new_tokens, num_samples, seed, output_dir):
    
    log.info("Starting Steering Validation")
    
    # 1. Load Model
    model_name = "gpt2" 
    log.info(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    log.info(f"Model loaded on {device}")
    
    # Initialize Scorer
    scorer = LexicalFieldScorer()
    # Pre-fetch lexical field to show user what we are looking for
    lex_field = scorer.get_lexical_field(concept)
    log.info(f"Lexical field for '{concept}': {lex_field}")

    # Define range of coefficients
    coeffs = np.linspace(coeff_start, coeff_end, coeff_steps)
    
    base_path = pathlib.Path(vector_base_dir)
    results_all = []

    # Loop over all configurations
    for c_type in concept_types: # e.g. cot_lat
        type_path = base_path / c_type
        
        # Find latest timestamp
        latest_dir = get_latest_timestamp_dir(type_path)
        if latest_dir is None:
            log.warning(f"No results found for type {c_type} in {type_path}")
            continue
        log.info(f"Using results from {latest_dir} for {c_type}")

        for l_type in layer_types: # e.g. attn.c_proj
            for layer_idx in layer_indices:
                # Construct full layer name e.g. transformer.h.6.attn.c_proj
                # Only support gpt2 style naming for now as per codebase convention
                full_layer_name = f"transformer.h.{layer_idx}.{l_type}"
                
                for suffix in vector_suffixes: # e.g. masked
                    # Construct vector filename
                    # vectors_transformer.h.6.attn.c_proj_masked.pkl
                    vector_filename = f"vectors_{full_layer_name}_{suffix}.pkl"
                    vector_path = latest_dir / vector_filename
                    
                    if not vector_path.exists():
                        log.warning(f"Vector file not found: {vector_path}")
                        continue

                    log.info(f"Steering with: Type={c_type}, Layer={full_layer_name}, Suffix={suffix}")

                    # Load Vector
                    try:
                        vector_np, max_act = load_vector(vector_path, concept)
                        vector = torch.tensor(vector_np).to(device)
                    except Exception as e:
                        log.error(f"Failed to load vector from {vector_path}: {e}")
                        continue

                    # Locate Target Module
                    target_module = None
                    for name, module in model.named_modules():
                        if name == full_layer_name:
                            target_module = module
                            break
                    
                    if target_module is None:
                        log.error(f"Layer {full_layer_name} not found in model")
                        continue

                    # Steering Loop
                    for coeff in tqdm(coeffs, desc=f"Steering {c_type} {full_layer_name} {suffix}"):
                        # Register hook
                        hook = SteeringHook(vector, coeff, max_act=max_act)
                        handle = target_module.register_forward_hook(hook)
                        
                        generated_texts = []
                        prompt_scores = []
                        
                        torch.manual_seed(seed) # Reset seed for consistent baseline per coeff

                        for prompt_tpl in NEUTRAL_PROMPTS:
                            # Some neutral prompts might not need format, but if they do:
                            try:
                                prompt = prompt_tpl.format(concept=concept)
                            except KeyError:
                                prompt = prompt_tpl

                            for _ in range(num_samples):
                                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                                with torch.no_grad():
                                    output = model.generate(
                                        input_ids, 
                                        max_new_tokens=max_new_tokens, 
                                        do_sample=True, 
                                        temperature=0.9,
                                        pad_token_id=tokenizer.eos_token_id
                                    )
                                text = tokenizer.decode(output[0], skip_special_tokens=True)
                                generated_texts.append(text)
                                
                                # Score
                                s = scorer.score(text, concept)
                                prompt_scores.append(s)

                        handle.remove()
                        
                        # Average score for this coefficient
                        avg_score = sum(prompt_scores) / len(prompt_scores) if prompt_scores else 0.0
                        
                        results_all.append({
                            "method": c_type,
                            "layer": full_layer_name,
                            "vector_type": suffix,
                            "factor": coeff,
                            "score": avg_score,
                            "texts": generated_texts[:2] # Save first 2 texts to keep CSV smallish 
                        })

    # Save aggregated results
    if not results_all:
        log.warning("No results generated!")
        return

    output_path = pathlib.Path(output_dir)
    df = pd.DataFrame(results_all)
    csv_path = output_path / f"steering_validation_{concept}_aggregated.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Saved aggregated results to {csv_path}")

    # Plotting - IEEE Style Comparison
    # Strategy: One plot file per Concept Type (LAT, PCA) to keep things manageable.
    # Within each file, we create a compact grid. 
    # To respect "max 2 subplots" constraint per view, we will group by Layer Type.
    
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.1, rc={"lines.linewidth": 1.5})
    
    # 1. Check if we have single point (steps=1) -> Use bar or scatter
    is_single_point = len(df["factor"].unique()) == 1
    
    unique_methods = df["method"].unique()
    
    for method in unique_methods:
        method_df = df[df["method"] == method]
        if method_df.empty: 
            continue
            
        # We will create a figure with columns = Layer Types (e.g. attn.c_proj, mlp.c_fc)
        # This keeps it to ~2 columns usually.
        # We will separate layers (5,6,7) by Style (dashes) or just aggregate them if too messy.
        # Let's use Style for Layer Index.
        
        # Extract layer index from full name for cleaner legend
        # e.g. transformer.h.6.attn.c_proj -> 6
        def get_layer_idx(name):
            try:
                return name.split('.')[2]
            except:
                return name
        
        method_df = method_df.copy()
        method_df["layer_idx"] = method_df["layer"].apply(get_layer_idx)
        
        # Extract layer type for column
        # e.g. transformer.h.6.attn.c_proj -> attn.c_proj
        def get_layer_type(name):
            if "attn" in name: return "Attention"
            if "mlp" in name: return "MLP"
            return "Layer"
        
        method_df["layer_group"] = method_df["layer"].apply(get_layer_type)

        if is_single_point:
            # Barplot for single comparison point - Use catplot to handle hue properly (side-by-side bars)
            # Group by Layer Index on X-axis to compare specific layers
            g = sns.catplot(data=method_df, kind="bar",
                            x="layer_idx", y="score", hue="vector_type",
                            col="layer_group", col_wrap=2,
                            height=5, aspect=1.2, palette="deep",
                            alpha=0.8, errorbar=None) # No error bar for single point usually, or 'sd' if multiple samples
            g.set_axis_labels("Layer Index", "Lexical Overlap Score")
        else:
            # Lineplot for curves - Use relplot
            # Aggregates over "layer_idx" (multiple lines -> mean + band) automatically
            g = sns.relplot(data=method_df, kind="line",
                            x="factor", y="score", hue="vector_type",
                            col="layer_group", col_wrap=2,
                            height=5, aspect=1.2, palette="deep",
                            marker="o", markersize=6) # err_style='band' is default for line
            g.set_axis_labels("Steering Factor", "Lexical Overlap Score")

        g.figure.suptitle(f"{concept} - {method.upper()}", y=1.05)
        
        plot_path = output_path / f"steering_plot_{concept}_{method}_compact.png"
        g.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        log.info(f"Saved compact plot to {plot_path}")



@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Retrieve lists from config (Omegaconf lists act like python lists)
    run_steering_validation(
        vector_base_dir=cfg.task.vector_base_dir,
        concept=cfg.task.concept,
        concept_types=cfg.task.concept_types,
        layer_indices=cfg.task.layer_indices,
        layer_types=cfg.task.layer_types,
        vector_suffixes=cfg.task.vector_suffixes,
        coeff_start=cfg.task.get("coeff_start", -5.0),
        coeff_end=cfg.task.get("coeff_end", 20.0),
        coeff_steps=cfg.task.get("coeff_steps", 20),
        max_new_tokens=cfg.task.max_new_tokens,
        num_samples=cfg.task.get("num_samples", 3),
        seed=cfg.task.get("seed", 42),
        output_dir=os.getcwd()
    )

if __name__ == "__main__":
    main()
