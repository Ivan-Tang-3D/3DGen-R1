import os
os.environ['SPCONV_ALGO'] = 'native'
import torch
import uuid
import numpy as np
from PIL import Image
import trimesh
import open3d as o3d
import imageio
from huggingface_hub import hf_hub_download
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, TextIteratorStreamer
from qwen_vl_utils import process_vision_info
from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.models.sparse_structure_vqvae import VQVAE3D
from threading import Thread
import json

# === Utilities ===
def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _save_video_safe(video_path, frames, fps=15):
    """Save frames to an mp4 robustly. Falls back to alternative codecs or GIF.

    Args:
        video_path: target .mp4 path
        frames: list of HxWxC uint8 arrays
        fps: frames per second
    Returns: final saved path (may be .gif fallback)
    """
    _ensure_dir(video_path)
    try:
        # Prefer imageio-ffmpeg writer with explicit codec to avoid 'codec_name/template' error
        with imageio.get_writer(video_path, fps=fps, codec='libx264', format='FFMPEG') as writer:
            for f in frames:
                writer.append_data(f)
        return video_path
    except Exception:
        # Try mpeg4 codec
        try:
            with imageio.get_writer(video_path, fps=fps, codec='mpeg4', format='FFMPEG') as writer:
                for f in frames:
                    writer.append_data(f)
            return video_path
        except Exception:
            # Fallback to GIF to avoid blocking pipeline
            gif_path = os.path.splitext(video_path)[0] + '.gif'
            try:
                imageio.mimsave(gif_path, frames, duration=1.0/float(max(fps,1)))
                return gif_path
            except Exception:
                # Give up on video; continue without raising
                return None

# Mesh to token utilities
def load_vertices(filepath):
    mesh = trimesh.load(filepath, force='mesh')
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces, dtype=np.int32))
    vertices = np.asarray(o3d_mesh.vertices)
    min_vals = vertices.min()
    max_vals = vertices.max()
    vertices_normalized = (vertices - min_vals) / (max_vals - min_vals)
    vertices = vertices_normalized * 1.0 - 0.5
    vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5)
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    vertices = (vertices + 0.5) / 64 - 0.5
    return vertices

def mesh_to_token_words(vqvae, mesh_path, device):
    verts = load_vertices(mesh_path)
    coords = ((torch.from_numpy(verts) + 0.5) * 64).int().contiguous()
    ss = torch.zeros(1, 64, 64, 64, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    token = vqvae.Encode(ss.to(dtype=torch.float32).unsqueeze(0).to(device))
    token = token[0].cpu().numpy().tolist()
    mesh = "<mesh-start>"
    for j in range(1024):
        mesh += f"<mesh{token[j]}>"
    mesh += "<mesh-end>"
    return mesh

class Inference:
    def __init__(self, model_dir="IvanTang/3DGen-R1", device="cuda"):
        self.device = torch.device(device)
        self.tmp_dir = "/tmp/3DGen-R1-demo"
        os.makedirs(self.tmp_dir, exist_ok=True)
        
        # Load VQVAE
        print("Loading VQVAE...")
        self.vqvae = VQVAE3D(num_embeddings=8192)
        self.vqvae.eval()
        filepath = hf_hub_download(repo_id="yejunliang23/3DVQVAE", filename="3DVQVAE.bin")
        state_dict = torch.load(filepath, map_location="cpu")
        self.vqvae.load_state_dict(state_dict)
        self.vqvae = self.vqvae.to(self.device)
        
        # Load 3DGen-R1 model
        print("Loading 3DGen-R1 model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir, torch_dtype="auto", device_map={"": 0}
        )
        self.model = self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.tokenizer = self.processor.tokenizer
        
        # Load Trellis pipelines
        print("Loading Trellis pipelines...")
        self.pipeline_text = TrellisTextTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-text-xlarge")
        self.pipeline_text.to(self.device)
        self.pipeline_image = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline_image.to(self.device)
        
        print("All models loaded successfully!")
    
    def _transform_messages(self, original_messages):
        """Transform messages to Qwen format"""
        transformed_messages = []
        for message in original_messages:
            new_content = []
            for item in message['content']:
                if 'image' in item:
                    new_content.append({'type': 'image', 'image': item['image']})
                elif 'text' in item:
                    new_content.append({'type': 'text', 'text': item['text']})
                elif 'video' in item:
                    new_content.append({'type': 'video', 'video': item['video']})
            if new_content:
                transformed_messages.append({'role': message['role'], 'content': new_content})
        return transformed_messages
    
    def _token_to_mesh(self, response_text):
        """Extract mesh tokens from response text"""
        tokens = []
        parts = response_text.split("><mesh")
        for part in parts:
            try:
                if part.startswith("<mesh"):
                    tokens.append(int(part[5:]))
                else:
                    tokens.append(int(part))
            except (ValueError, IndexError):
                continue
        
        # Pad to 1024 tokens
        while len(tokens) < 1024:
            tokens.append(tokens[-1] if tokens else 0)
        
        return torch.tensor(tokens).unsqueeze(0)
    
    def _move_to_device(self, inputs):
        """Move all tensors in inputs to device"""
        device_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                device_inputs[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_inputs[key] = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                     for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                processed = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        processed.append(item.to(self.device))
                    elif isinstance(item, dict):
                        processed.append({k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                         for k, v in item.items()})
                    else:
                        processed.append(item)
                device_inputs[key] = type(value)(processed)
            else:
                device_inputs[key] = value
        return device_inputs
    
    def _generate_tokens(self, messages, gen_params):
        """Generate tokens from messages"""
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, 
                               padding=True, return_tensors='pt')
        device_inputs = self._move_to_device(inputs)
        
        eos_token_id = [self.tokenizer.eos_token_id, 159858]
        streamer = TextIteratorStreamer(self.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        
        gen_kwargs = {
            'max_new_tokens': gen_params.get('max_new_tokens', 2048),
            'streamer': streamer,
            'eos_token_id': eos_token_id,
            'top_k': gen_params.get('top_k', 8192),
            'top_p': gen_params.get('top_p', 0.7),
            'temperature': gen_params.get('temperature', 0.7),
            **device_inputs
        }
        
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        full_response = ""
        encoding_indices = None
        
        for new_text in streamer:
            if new_text:
                if "<mesh" in new_text:
                    encoding_indices = self._token_to_mesh(new_text)
                full_response += new_text
                print(f"Generated: {new_text}", end="", flush=True)
        
        print(f"\nFull response: {full_response}")
        return full_response, encoding_indices
    
    def _decode_mesh(self, encoding_indices):
        """Decode mesh tokens to coordinates"""
        recon = self.vqvae.Decode(encoding_indices.to(self.device))
        z_s = (recon[0].detach().cpu() > 0).float()
        indices = torch.nonzero(z_s[0] == 1)
        position_recon = (indices.float() + 0.5) / 64 - 0.5
        
        coords = ((position_recon + 0.5) * 64).int().contiguous()
        ss = torch.zeros(1, 64, 64, 64, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        ss = ss.unsqueeze(0)
        coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()
        return coords.to(self.device)
    
    def _save_outputs(self, outputs, simplify, texture_size, save_path=None):
        """Save generated 3D model and video"""
        trial_id = uuid.uuid4()
        video_path = f"{self.tmp_dir}/{trial_id}.mp4"
        
        # Render video
        video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
        video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
        video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
        saved_video_path = _save_video_safe(video_path, video, fps=15)
        
        # Export GLB
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            simplify=simplify,
            texture_size=texture_size,
            verbose=False
        )
        
        glb_path = save_path if save_path else f"{self.tmp_dir}/{trial_id}.glb"
        _ensure_dir(glb_path)
        glb.export(glb_path)
        
        print(f"3D model saved to: {glb_path}")
        if saved_video_path:
            print(f"Video saved to: {saved_video_path}")
        
        return glb_path, saved_video_path
    
    def generate_3d_from_text(self, prompt, seed=42, top_k=8192, top_p=0.7, temperature=0.7, 
                             simplify=0.95, texture_size=1024, save_path=None):
        """Generate 3D model from text prompt"""
        torch.manual_seed(seed)
        
        messages = [{'role': 'user', 'content': [{'text': prompt}]}]
        messages = self._transform_messages(messages)
        
        gen_params = {'top_k': top_k, 'top_p': top_p, 'temperature': temperature}
        full_response, encoding_indices = self._generate_tokens(messages, gen_params)
        
        if encoding_indices is None:
            return {"error": "No mesh tokens generated"}
        
        print("Processing mesh...")
        coords = self._decode_mesh(encoding_indices)
        
        try:
            with torch.no_grad():
                cond = self.pipeline_text.get_cond([prompt])
                slat = self.pipeline_text.sample_slat(cond, coords)
                outputs = self.pipeline_text.decode_slat(slat, ['mesh', 'gaussian'])
            
            glb_path, video_path = self._save_outputs(outputs, simplify, texture_size, save_path)
            
            return {
                "success": True,
                "glb_path": glb_path,
                "video_path": video_path,
                "prompt": prompt,
                "response": full_response
            }
        except Exception as e:
            print(f"Error during 3D generation: {e}")
            return {"error": str(e)}


def main():
    """Example usage"""
    print("Initializing 3DGen-R1 inference...")
    inference = Inference()
    
    # Text to 3D
    print("\n=== Text to 3D Example ===")
    text_prompt = "A futuristic car with four wheels and a streamlined body."
    save_dir = "./test_multimodal_outputs"
    os.makedirs(save_dir, exist_ok=True)
    mesh_path = os.path.join(save_dir, "test_output.glb")
    result_3d = inference.generate_3d_from_text(prompt=text_prompt, save_path=mesh_path)
    print(f"3D generation result: {result_3d}")
    
    # Save results
    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"text_to_3d": result_3d}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main() 