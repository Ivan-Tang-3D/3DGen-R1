import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T
import clip
import json
import csv
import argparse
import multiprocessing as mp
import traceback
from sklearn.metrics.pairwise import rbf_kernel
# DINOv2
from transformers import AutoImageProcessor, Dinov2Model

# Detect --allow_online flag before argparse
def _detect_allow_online_from_argv() -> bool:
    try:
        argv = sys.argv
        for i, a in enumerate(argv):
            if a == "--allow_online":
                if i + 1 < len(argv):
                    v = argv[i + 1].strip().lower()
                    return v in ("1", "true", "yes")
                return True
    except Exception:
        pass
    # Default to True if not explicitly set
    env_val = os.environ.get("ALLOW_ONLINE", "1")
    return env_val.lower() in ("1", "true", "yes")

# Model cache directory
MODEL_CACHE_DIR = os.environ.get('MODEL_CACHE_DIR', '/YOUR/MODEL/CACHE/PATH')

def setup_local_models(allow_online: bool = False):
    """Setup local model loading to avoid network downloads"""
    os.environ['HF_HUB_CACHE'] = MODEL_CACHE_DIR
    os.environ['TRANSFORMERS_CACHE'] = MODEL_CACHE_DIR
    os.environ['CLIP_CACHE_DIR'] = MODEL_CACHE_DIR
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    
    if not allow_online:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # Patch hf_hub_download to check local cache first, then allow download
    try:
        import huggingface_hub
        original_hf_hub_download = huggingface_hub.hf_hub_download
        
        def custom_hf_hub_download(repo_id, filename, **kwargs):
            # Check for 3DVQVAE in local cache
            if repo_id == "yejunliang23/3DVQVAE" and filename == "3DVQVAE.bin":
                local_path = os.path.join(MODEL_CACHE_DIR, "3DVQVAE", "3DVQVAE.bin")
                if os.path.exists(local_path):
                    print(f"Using local 3DVQVAE: {local_path}")
                    return local_path
                # If not found locally and online is allowed, download
                if allow_online:
                    print(f"3DVQVAE not found locally, downloading from HuggingFace...")
                    kwargs.pop('local_files_only', None)
                else:
                    kwargs['local_files_only'] = True
            return original_hf_hub_download(repo_id, filename, **kwargs)
        
        huggingface_hub.hf_hub_download = custom_hf_hub_download
        # Also patch transformers if available
        try:
            import transformers
            transformers.hf_hub_download = custom_hf_hub_download
        except:
            pass
    except Exception as e:
        print(f"Warning: Failed to patch hf_hub_download: {e}")


    # Patch CLIP to use local model
    try:
        original_clip_load = clip.load
        def custom_clip_load(name, device="cpu", download_root=None):
            if name == "ViT-B/32":
                local_clip_path = os.path.join(MODEL_CACHE_DIR, "clip", "ViT-B-32.pt")
                if os.path.exists(local_clip_path):
                    return original_clip_load(name, device, os.path.dirname(local_clip_path))
            return original_clip_load(name, device, download_root)
        clip.load = custom_clip_load
    except Exception as e:
        print(f"Warning: Failed to patch clip.load: {e}")

# Setup local model loading
_ALLOW_ONLINE = _detect_allow_online_from_argv()
os.environ['ALLOW_ONLINE'] = '1' if _ALLOW_ONLINE else '0'
setup_local_models(allow_online=_ALLOW_ONLINE)

sys.path.append(os.path.dirname(__file__))
from inference import Inference

# Configuration
MODEL_PATH = "/YOUR/MODEL/PATH"
METADATA_CSV = "./example/metadata_example.csv"
RENDERS_ROOT = "./example"
EVAL_LIST = "./example/example.csv"
OUTPUT_DIR = "./eval_outputs"
EVAL_ROOT = "./eval_outputs/eval_reg_results"
NUM_VIEWS = 24
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EVAL_ROOT, exist_ok=True)

def render_mesh_to_images(glb_path, transforms_json_path, out_size=299):
    """Use pyrender-only multi-view rendering (no Open3D rendering dependency)."""
    with open(transforms_json_path, 'r') as f:
        transforms = json.load(f)
    frames = transforms['frames']

    import pyrender
    import trimesh as _tm

    tm_mesh = _tm.load(glb_path, force='mesh')
    if not isinstance(tm_mesh, _tm.Trimesh) and hasattr(tm_mesh, 'dump'):
        tm_mesh = _tm.util.concatenate(tm_mesh.dump())

    material = pyrender.MetallicRoughnessMaterial(baseColorFactor=[1.0, 1.0, 1.0, 1.0])
    pr_mesh = pyrender.Mesh.from_trimesh(tm_mesh, material=material, smooth=True)

    images = []
    r = pyrender.OffscreenRenderer(viewport_width=out_size, viewport_height=out_size)
    try:
        for frame in frames:
            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0])
            scene.add(pr_mesh)
            m = np.array(frame['transform_matrix'])
            eye = m[:3, 3]
            center = eye - m[:3, 2]
            up = m[:3, 1]
            # Camera
            camera_angle_x = frame.get('camera_angle_x', None)
            if camera_angle_x is not None:
                yfov = float(camera_angle_x)
                cam = pyrender.PerspectiveCamera(yfov=yfov)
            else:
                cam = pyrender.PerspectiveCamera(yfov=np.deg2rad(60))
            cam_node = scene.add(cam, pose=_look_at(eye, center, up))
            # Light
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
            scene.add(light, pose=_look_at(eye, center, up))
            color, _ = r.render(scene)
            images.append(Image.fromarray(color))
            scene.remove_node(cam_node)
    finally:
        r.delete()
    return images

def _look_at(eye, center, up):
    eye = np.asarray(eye, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    z = eye - center
    z = z / (np.linalg.norm(z) + 1e-8)
    x = np.cross(up, z)
    x = x / (np.linalg.norm(x) + 1e-8)
    y = np.cross(z, x)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = x
    m[:3, 1] = y
    m[:3, 2] = z
    m[:3, 3] = eye
    return m

def get_clip_score(images, prompt, clip_model, clip_preprocess, device):
    imgs = torch.stack([clip_preprocess(img.convert('RGB')) for img in images]).to(device)
    if isinstance(prompt, str):
        short_prompt = prompt.split('.')[0]
        if len(short_prompt.strip()) == 0:
            short_prompt = prompt[:77]
    else:
        short_prompt = str(prompt)[:77]
    texts = clip.tokenize([short_prompt] * len(images)).to(device)
    with torch.no_grad():
        img_feats = clip_model.encode_image(imgs)
        text_feats = clip_model.encode_text(texts)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
        scores = (img_feats @ text_feats.T).cpu().numpy()
    return scores.diagonal().mean()

def get_dinov2_features(images, processor, model, device):
    imgs = [img.convert('RGB').resize((224, 224)) for img in images]
    inputs = processor(images=imgs, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        feats = outputs.last_hidden_state[:, 0]
    return feats.cpu().numpy()

def compute_kd(feats_real, feats_fake):
    """Compute Kernel Distance using RBF kernel"""
    feats_real = np.asarray(feats_real)
    feats_fake = np.asarray(feats_fake)
    
    if len(feats_real.shape) == 1:
        feats_real = feats_real.reshape(1, -1)
    if len(feats_fake.shape) == 1:
        feats_fake = feats_fake.reshape(1, -1)
    
    kernel_matrix = rbf_kernel(feats_real, feats_fake)
    return np.mean(kernel_matrix)

def process_single_sample(args):
    sha256, prompt, device_id = args
    try:
        if device_id >= 0 and torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
            device = "cuda:0"
        else:
            device = "cpu"
    except Exception as e:
        print(f"Warning: Failed to set CUDA device {device_id}, using CPU: {e}")
        device = "cpu"
    
    try:
        # Load models
        inference = Inference(model_dir=MODEL_PATH, device=device)
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        dinov2_model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()

        gt_dir = os.path.join(RENDERS_ROOT, sha256)
        transforms_json_path = os.path.join(gt_dir, "transforms.json")
        gen_mesh_path = os.path.join(OUTPUT_DIR, f"{sha256}.glb")

        # Load GT images
        try:
            gt_imgs = [Image.open(os.path.join(gt_dir, f"{i:03d}.png")) for i in range(NUM_VIEWS)]
        except Exception as e:
            print(f"[Warning] Failed to load GT images for {sha256}: {e}, skipping.")
            return None

        # Generate 3D model
        try:
            result = inference.generate_3d_from_text(
                prompt=f"Please generate a 3D mesh based on the prompt I provided: {prompt}",
                save_path=gen_mesh_path,
                top_k=8192, top_p=0.7, temperature=0.7
            )
            if not result.get("success") or not os.path.exists(result.get("glb_path", "")):
                print(f"[Warning] Failed to generate or save mesh for: {sha256}, skipping.")
                return None
        except Exception as e:
            print(f"[Warning] Exception during mesh generation for {sha256}: {e}, skipping.")
            return None

        # Render generated model
        try:
            gen_imgs = render_mesh_to_images(result["glb_path"], transforms_json_path, out_size=299)
            if len(gen_imgs) != NUM_VIEWS:
                print(f"[Warning] Rendered image count {len(gen_imgs)} != NUM_VIEWS for {sha256}, skipping.")
                return None
        except Exception as e:
            print(f"[Warning] Rendering failed for {sha256}: {e}, skipping.")
            return None

        # Compute metrics
        try:
            clip_score = get_clip_score(gen_imgs, prompt, clip_model, clip_preprocess, device)
            gt_dinov2_feats = get_dinov2_features(gt_imgs, dinov2_processor, dinov2_model, device)
            gen_dinov2_feats = get_dinov2_features(gen_imgs, dinov2_processor, dinov2_model, device)
            dinov2_kd = compute_kd(gt_dinov2_feats, gen_dinov2_feats)
            if np.isnan(dinov2_kd) or np.isinf(dinov2_kd):
                dinov2_kd = 0.0
        except Exception as e:
            print(f"[Warning] Metric computation failed for {sha256}: {e}, skipping.")
            return None

        return [sha256, prompt, clip_score, dinov2_kd]
    except Exception as e:
        print(f"[Error] Unexpected error for {sha256}: {e}")
        traceback.print_exc()
        return None

def main():
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--sha256', type=str, default=None, help='Evaluate specific sha256 sample')
    parser.add_argument('--num_processes', type=int, default=1, help='Number of parallel processes')
    parser.add_argument('--allow_online', type=int, default=1 if _ALLOW_ONLINE else 0, help='Allow online download (1) or strict offline (0)')
    args = parser.parse_args()

    os.environ['ALLOW_ONLINE'] = '1' if int(args.allow_online) == 1 else '0'

    # Load test samples
    eval_csv_path = EVAL_LIST
    if os.path.exists(eval_csv_path):
        print(f"Reading test samples from {eval_csv_path}")
        eval_df = pd.read_csv(eval_csv_path)
        test_sha256s = eval_df['sha256'].tolist()
        print(f"Selected {len(test_sha256s)} test samples from list")
        
        df = pd.read_csv(METADATA_CSV)
        df = df[df['sha256'].isin(test_sha256s)]
        print(f"Found {len(df)} matching samples in metadata")
    else:
        print(f"Warning: {eval_csv_path} not found, using original metadata")
        df = pd.read_csv(METADATA_CSV)

    if args.sha256 is not None:
        df = df[df['sha256'] == args.sha256]
        if len(df) == 0:
            print(f"sha256 {args.sha256} not found in metadata.")
            return

    csv_path = os.path.join(EVAL_ROOT, "eval_icp.csv")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args_list = []
    for idx, row in df.iterrows():
        if num_gpus == 0:
            device_id = -1
        else:
            device_id = 0 if args.num_processes == 1 else (idx % num_gpus)
        args_list.append((row['sha256'], row['captions'], device_id))

    results = []
    with mp.Pool(processes=args.num_processes) as pool:
        for res in tqdm(pool.imap(process_single_sample, args_list), total=len(args_list)):
            if res is not None:
                results.append(res)
                with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    if write_header:
                        writer.writerow(["sha256", "prompt", "clip_score", "dinov2_kd"])
                        write_header = False
                    writer.writerow(res)

    # Print mean metrics
    if results:
        arr = np.array([[float(x) if x is not None else np.nan for x in r[2:]] for r in results])
        print(f"Mean CLIP-SCORE: {np.nanmean(arr[:,0]):.4f}")
        print(f"Mean DINOv2 KD: {np.nanmean(arr[:,1]):.4f}")
    else:
        print("No successful evaluations.")

if __name__ == "__main__":
    main()