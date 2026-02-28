import torch
import numpy as np
from typing import List, Dict, Any, Optional
import re
from abc import ABC, abstractmethod
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import trimesh
import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.models.sparse_structure_vqvae import VQVAE3D


class Base3DRewardModel(ABC):
    
    def __init__(self, config=None):
        self.config = config or {}
        self.device = None
        
    @abstractmethod
    def compute_reward(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        
        Args:
            data: prompts, responses, mesh_tokens dict
            
        Returns:
            torch.Tensor: reward score
        """
        pass
    
    def load_to_device(self, device):
        self.device = device


class VLM3DReward(Base3DRewardModel):
    
    def __init__(self, config=None):
        super().__init__(config)
        if hasattr(config, '__dict__'): 
            self.model_name = getattr(config, 'vlm_reward_model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
            self.temperature = getattr(config, 'vlm_reward_temperature', 0.0)
            self.joint_multiview = getattr(config, 'vlm_reward_joint_multiview', False)
            self.strict_all_views = getattr(config, 'vlm_reward_strict_all_views', False)
        else:  
            config_dict = config or {}
            self.model_name = config_dict.get('model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
            self.temperature = config_dict.get('temperature', 0.0)
            self.joint_multiview = config_dict.get('vlm_reward_joint_multiview', False)
            self.strict_all_views = config_dict.get('vlm_reward_strict_all_views', False)
      
        self.model = None
        self.processor = None
        self.accelerator = None
        self._deepspeed_enabled = False
        
        self.reward_prompt_template = """You are given a text prompt: "{prompt}"
Below is one generated 3D model visualization: <image>

1. Describe the 3D model thoroughly (shape, structure, geometry, materials, etc.), do not be affected by the prompt.
2. Identify key 3D elements and instructions from the prompt.
3. Evaluate how well the 3D model follows the prompt:
   - Are all required 3D elements present?
   - Are object shapes, proportions, and spatial relationships accurate?
   - Does the 3D model match the described geometry and materials?

Be extremely strict and precise:
Only if the 3D model matches the prompt perfectly, respond with: \\boxed{{1}}.
Otherwise, respond with: \\boxed{{0}}.

Reason before your final boxed answer. Only one number should appear inside the box."""
        
    @property
    def __name__(self):
        return 'VLM3DReward'
    
    def load_to_device(self, device):
        """加载VLM模型到指定设备"""
        self.device = device
        if self.model is None:
            self._load_model()
    
    def _load_model(self):
        """提取模型加载逻辑"""
        print(f"Loading VLM reward model: {self.model_name} on device: {self.device}")
        
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=None,  
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        
        if self.device is not None:
            print(f"Moving reward model to {self.device}")
            self.model = self.model.to(self.device)
        print(f"✅ VLM reward model loaded successfully on {self.device}")
    
    def _preprocess_views(self, views) -> List[Image.Image]:
        view_images = []
        if not isinstance(views, list):
            return view_images
        
        for i, v in enumerate(views):
            try:
                if isinstance(v, Image.Image):
                    view_images.append(v)
                elif isinstance(v, np.ndarray):
                    if v.ndim == 3 and v.shape[2] in [3, 4]:  
                        if v.dtype != np.uint8:
                            if v.max() <= 1.0:
                                v = (v * 255).astype(np.uint8)
                            else:
                                v = v.astype(np.uint8)
                        if v.shape[2] == 4:
                            v = v[:, :, :3]
                        view_images.append(Image.fromarray(v))
                    else:
                        print(f"Warning: Invalid array shape at index {i}: {v.shape}")
                else:
                    print(f"Warning: Unsupported view type at index {i}: {type(v)}")
            except Exception as e:
                print(f"Warning: Failed to process view {i}: {e}")
                
        return view_images

    def compute_reward(self, data: Dict[str, Any]) -> torch.Tensor:
        prompts = data.get('prompts', [])
        multi_view_images = data.get('multi_view_images', [])  # List[List[np.ndarray]]
        
        if self.model is None:
            self.load_to_device(self.device)
        
        scores = []
        for i, (prompt, views) in enumerate(zip(prompts, multi_view_images)):
            try:
                view_images = self._preprocess_views(views)

                if len(view_images) == 0:
                    reward_score = 0.0
                elif self.joint_multiview:
                    reward_score = float(self._compute_joint_multiview_reward(prompt, view_images))
                else:
                    per_view_scores = [self._compute_single_reward(prompt, img) for img in view_images]
                    if self.strict_all_views:
                        reward_score = 1.0 if all(s >= 1.0 for s in per_view_scores) else 0.0
                    else:
                        positives = sum(1 for s in per_view_scores if s >= 1.0)
                        reward_score = 1.0 if positives >= (len(per_view_scores) + 1) // 2 else 0.0
                    
                scores.append(reward_score)
                
            except Exception as e:
                print(f"Error computing VLM reward for sample {i}: {e}")
                scores.append(0.0)
        
        return torch.tensor(scores, dtype=torch.float32, device=self.device)
    
    def _compute_single_reward(self, prompt: str, image: Image.Image) -> float:
        try:
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                    image = Image.fromarray(image)
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")
                
            query_prompt = self.reward_prompt_template.format(prompt=prompt)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query_prompt},
                        {"type": "image", "image": image}
                    ]
                }
            ]
            
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.processor(
                text=[text], 
                images=[image], 
                return_tensors="pt", 
                padding=True
            )
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=False, 
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            generated_text = self.processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )[0]
            reward = self._extract_boxed_answer(generated_text)
            
            return float(reward)
            
        except Exception as e:
            print(f"Error in VLM reward computation: {e}")
            return 0.0

    def _compute_joint_multiview_reward(self, prompt: str, images: list[Image.Image]) -> int:
        try:
            query_prompt = (
                "You are given a text prompt: \"{prompt}\"\n"
                "Below are multiple views of one generated 3D object: <image_1> ... <image_N>\n\n"
                "1. Describe the 3D object considering all views jointly.\n"
                "2. Identify key elements from the prompt.\n"
                "3. Judge if the object matches the prompt across all views.\n\n"
                "Be extremely strict and precise:\n"
                "Only if the object matches the prompt perfectly across all views, respond with: \\boxed{{1}}.\n"
                "Otherwise, respond with: \\boxed{{0}}.\n\n"
                "Reason before your final boxed answer. Only one number should appear inside the box."
            ).format(prompt=prompt)

            contents = [{"type": "text", "text": query_prompt}]
            for img in images:
                contents.append({"type": "image", "image": img})

            messages = [{"role": "user", "content": contents}]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )

            generated_text = self.processor.batch_decode(
                generated_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )[0]
            return self._extract_boxed_answer(generated_text)
        except Exception as e:
            print(f"Error in joint multiview reward computation: {e}")
            return 0
    
    def _extract_boxed_answer(self, text: str) -> int:
        try:
            pattern = r'\\boxed\{(\d+)\}'
            matches = re.findall(pattern, text)
            
            if matches:
                return int(matches[-1])  
            else:
                numbers = re.findall(r'\b[01]\b', text)
                if numbers:
                    return int(numbers[-1])
                else:
                    return 0
                    
        except Exception as e:
            print(f"Error extracting boxed answer: {e}")
            return 0
    
    def _render_glb_to_image(self, glb_model) -> Optional[Image.Image]:
        try:
            if isinstance(glb_model, str):
                mesh = trimesh.load(glb_model, force='mesh')
            else:
                mesh = glb_model
            
            scene = mesh.scene()
            image = scene.save_image(resolution=(512, 512))
            
            if image is not None:
                return Image.fromarray(image)
            else:
                return None
                
        except Exception as e:
            print(f"Error rendering GLB to image: {e}")
            return None




# 奖励函数注册表
reward_funcs_registry_3d = {
    "vlm_3d": VLM3DReward,
}


def create_3d_reward_model(reward_type: str, config: Optional[Dict] = None) -> Base3DRewardModel:
    if reward_type not in reward_funcs_registry_3d:
        raise ValueError(f"Unknown reward type: {reward_type}. Available types: {list(reward_funcs_registry_3d.keys())}")
    
    return reward_funcs_registry_3d[reward_type](config)