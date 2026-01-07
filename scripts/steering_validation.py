import hydra
from omegaconf import DictConfig
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

# Add project root to path
sys.path.append(os.getcwd())

log = logging.getLogger(__name__)

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

class SentenceEmbeddingScorer:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        log.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def score(self, text, concept):
        # Embed text and concept
        emb_text = self.get_embedding(text)
        emb_concept = self.get_embedding(concept)
        
        # Cosine similarity
        sim = F.cosine_similarity(emb_text, emb_concept)
        return sim.item()

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
    
    # Remove suffix (_global, _masked, _killed)
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

def run_steering_validation(vector_path, concept, layer_name, prompt_template, 
                            coeff_start, coeff_end, coeff_steps, 
                            max_new_tokens, num_samples, seed, output_dir):
    
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
    
    # 2. Load Vector
    try:
        vector_np, max_act = load_vector(vector_path, concept)
        vector = torch.tensor(vector_np).to(device)
        log.info(f"Loaded vector for concept: {concept}, Max Act: {max_act:.2f}")
        
    except Exception as e:
        log.error(f"Failed to load vector: {e}")
        return

    # Validate layer mismatch
    vector_source_layer = extract_layer_from_vector_path(vector_path)
    if vector_source_layer and vector_source_layer != layer_name:
        log.warning(
            f"LAYER MISMATCH: Vector was computed from '{vector_source_layer}' "
            f"but steering is being applied to '{layer_name}'. "
            f"This may cause poor steering results. Consider using layer='{vector_source_layer}'."
        )

    # Initialize Scorer
    scorer = SentenceEmbeddingScorer()

    # 3. Register Hook
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break
            
    if target_module is None:
        log.error(f"Layer {layer_name} not found in model")
        return

    # 4. Generate & Score
    prompt = prompt_template.format(concept=concept)
    
    # Define range of coefficients
    coeffs = np.linspace(coeff_start, coeff_end, coeff_steps)
    
    results = []
    
    log.info(f"Running steering for concept '{concept}' on layer '{layer_name}'")
    log.info(f"Coefficients: {coeffs}")
    
    for coeff in tqdm(coeffs, desc="Steering factors"):
        # Register hook
        hook = SteeringHook(vector, coeff, max_act=max_act)
        handle = target_module.register_forward_hook(hook)
        
        # Generate samples
        generated_texts = []
        torch.manual_seed(seed) # Reset seed for fair comparison
        
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
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_texts.append(generated_text)
            
        handle.remove()
        
        # Compute Concept Score (Sentence Embedding Similarity)
        # Average similarity across samples
        scores = [scorer.score(text, concept) for text in generated_texts]
        score = sum(scores) / len(scores)
        
        log.info(f"Factor: {coeff:.2f}, Score: {score:.2f}")
        for i, text in enumerate(generated_texts[:2]): # Log first 2 samples for debugging
            log.info(f"  Sample {i}: {text}")
        
        results.append({
            "factor": coeff,
            "score": score,
            "texts": generated_texts
        })
        
    # Save results
    output_path = pathlib.Path(output_dir)
    df = pd.DataFrame(results)
    df.to_csv(output_path / f"steering_results_{concept}_{layer_name}.csv", index=False)
    
    # Plot Factor vs Score
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="factor", y="score", marker="o")
    plt.title(f"Steering Factor vs Concept Score\nConcept: {concept}, Layer: {layer_name}")
    plt.xlabel("Steering Factor")
    plt.ylabel("Concept Score (Embedding Similarity)")
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.1, 1.1)
    plt.savefig(output_path / f"steering_plot_{concept}_{layer_name}.png")
    plt.close()
    
    log.info(f"Results saved to {output_path}")

@hydra.main(config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    run_steering_validation(
        vector_path=cfg.task.vector_path,
        concept=cfg.task.get("concept", None),
        layer_name=cfg.task.layer,
        prompt_template=cfg.task.prompt,
        coeff_start=cfg.task.get("coeff_start", -5.0),
        coeff_end=cfg.task.get("coeff_end", 5.0),
        coeff_steps=cfg.task.get("coeff_steps", 21),
        max_new_tokens=cfg.task.max_new_tokens,
        num_samples=cfg.task.get("num_samples", 5),
        seed=cfg.task.get("seed", 42),
        output_dir=os.getcwd()
    )

if __name__ == "__main__":
    main()
