import os
import sys
import pandas as pd

# Model cache directory
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/YOUR/MODEL/CACHE/PATH')

def setup_local_models():
    """Setup local model loading"""
    os.environ['HF_HUB_CACHE'] = MODEL_CACHE_DIR
    os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR
    os.environ['CLIP_CACHE_DIR'] = MODEL_CACHE_DIR
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    
    # Patch hf_hub_download to check local cache first
    try:
        import huggingface_hub
        original_hf_hub_download = huggingface_hub.hf_hub_download
        
        def custom_hf_hub_download(repo_id, filename, **kwargs):
            if repo_id == "yejunliang23/3DVQVAE" and filename == "3DVQVAE.bin":
                local_path = os.path.join(MODEL_CACHE_DIR, "3DVQVAE", "3DVQVAE.bin")
                if os.path.exists(local_path):
                    print(f"Using local 3DVQVAE: {local_path}")
                    return local_path
            return original_hf_hub_download(repo_id, filename, **kwargs)
        
        huggingface_hub.hf_hub_download = custom_hf_hub_download
        try:
            import transformers
            transformers.hf_hub_download = custom_hf_hub_download
        except:
            pass
    except Exception as e:
        print(f"Warning: Failed to patch hf_hub_download: {e}")

setup_local_models()

sys.path.append(os.path.dirname(__file__))
from inference import Inference

# Configuration
MODEL_PATH = "/YOUR/MODEL/PATH"
EXAMPLE_CSV = "./example/example.csv"  
OUTPUT_DIR = "./demo_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Load example samples
    if not os.path.exists(EXAMPLE_CSV):
        print(f"Error: Example CSV file not found: {EXAMPLE_CSV}")
        return
    
    print(f"Reading examples from {EXAMPLE_CSV}")
    df = pd.read_csv(EXAMPLE_CSV)
    
    print(f"Available columns: {list(df.columns)}")
    
    sha256_col = None
    prompt_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if sha256_col is None and ('sha256' in col_lower or 'id' in col_lower or 'hash' in col_lower):
            sha256_col = col
        if prompt_col is None and ('caption' in col_lower or 'prompt' in col_lower or 'text' in col_lower or 'description' in col_lower):
            prompt_col = col
    
    if sha256_col is None:
        if len(df.columns) >= 1:
            sha256_col = df.columns[0]
            print(f"Using first column as sha256: {sha256_col}")
        else:
            print("Error: CSV has no columns")
            return
    
    if prompt_col is None:
        if len(df.columns) >= 2:
            prompt_col = df.columns[1]
            print(f"Using second column as prompt: {prompt_col}")
        elif len(df.columns) == 1:
            prompt_col = df.columns[0]
            sha256_col = None  
            print(f"Using single column as prompt: {prompt_col}")
        else:
            print("Error: CSV must have at least 1 column (prompt/caption)")
            return
    
    print(f"Using columns: sha256={sha256_col}, prompt={prompt_col}")
    print(f"Found {len(df)} examples to process")
    
    # Initialize inference model
    print(f"\nInitializing inference model from {MODEL_PATH}...")
    try:
        inference = Inference(model_dir=MODEL_PATH, device="cuda")
    except Exception as e:
        print(f"Error: Failed to initialize inference model: {e}")
        return
    
    # Process each example
    for idx, row in df.iterrows():
        sha256 = str(row[sha256_col]) if sha256_col else f"sample_{idx+1}"
        prompt_raw = row[prompt_col]
        
        if isinstance(prompt_raw, list):
            prompt = str(prompt_raw[0]) if len(prompt_raw) > 0 else str(prompt_raw)
        elif isinstance(prompt_raw, str):
            if prompt_raw.strip().startswith('[') and prompt_raw.strip().endswith(']'):
                try:
                    import ast
                    prompt_list = ast.literal_eval(prompt_raw)
                    if isinstance(prompt_list, list) and len(prompt_list) > 0:
                        prompt = str(prompt_list[0])
                    else:
                        prompt = prompt_raw
                except:
                    prompt = prompt_raw
            else:
                prompt = prompt_raw
        else:
            prompt = str(prompt_raw)
        
        formatted_prompt = f"Please generate a 3D mesh based on the prompt I provided: {prompt}"
        
        print(f"\n[{idx+1}/{len(df)}] Processing {sha256}")
        print(f"Original prompt: {prompt}")
        
        filename = sha256 if sha256_col else f"sample_{idx+1}"
        save_path = os.path.join(OUTPUT_DIR, f"{filename}.glb")
        
        try:
            result = inference.generate_3d_from_text(
                prompt=formatted_prompt,
                save_path=save_path,
                top_k=8192,
                top_p=0.7,
                temperature=0.7
            )
            
            if result.get("success"):
                print(f"✓ Successfully generated: {save_path}")
            else:
                print(f"✗ Generation failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"✗ Exception: {e}")
    
    print(f"All results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
