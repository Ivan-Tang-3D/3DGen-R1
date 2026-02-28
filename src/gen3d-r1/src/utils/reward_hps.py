import os
import torch
import requests
from PIL import Image
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
import numpy as np
from typing import List, Dict, Any, Optional

class HPSv2:
    def __init__(self, args):
        self.ckpt_path = args.hps_ckpt_path

    @property
    def __name__(self):
        return 'HPSv2'
    
    def load_to_device(self, load_device):
        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
                    'ViT-H-14',
                    pretrained='laion2b_s32b_b79k',
                    precision='amp',
                    device='cuda',
                    jit=False,
                    force_quick_gelu=False,
                    force_custom_text=False,
                    force_patch_dropout=False,
                    force_image_size=None,
                    pretrained_image=False,
                    image_mean=None,
                    image_std=None,
                    light_augmentation=True,
                    aug_cfg={},
                    output_dict=True,
                    with_score_predictor=False,
                    with_region_predictor=False
                )
        # workaround for the zero3
        checkpoint = torch.load(self.ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.tokenizer = get_tokenizer('ViT-H-14')
        self.model = self.model.to(load_device)
        self.model.eval()
        print(f"HPSv2 loaded to {load_device}")
    
    def _preprocess_views(self, views):
        """预处理多视角图像"""
        view_images = []
        if not isinstance(views, list):
            return view_images
        
        for i, v in enumerate(views):
            try:
                if isinstance(v, Image.Image):
                    view_images.append(v)
                elif isinstance(v, np.ndarray):
                    # 确保数组格式正确
                    if v.ndim == 3 and v.shape[2] in [3, 4]:  # RGB或RGBA
                        if v.dtype != np.uint8:
                            if v.max() <= 1.0:
                                v = (v * 255).astype(np.uint8)
                            else:
                                v = v.astype(np.uint8)
                        # 如果是RGBA，转换为RGB
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
    
    def compute_reward(self, data):
        # image_list is a list of PIL image
        device = list(self.model.parameters())[0].device
        prompts = data.get('prompts', [])
        multi_view_images = data.get('multi_view_images', [])
        result = []
        
        for i, (prompt, views) in enumerate(zip(prompts, multi_view_images)):
            max_hps_score = 0.0
            #reward_hps_list = []
            view_images = self._preprocess_views(views)
            for image in view_images:
                with torch.no_grad():
                    image = self.preprocess_val(image).unsqueeze(0).to(device=device, non_blocking=True)
                    # Process the prompt
                    text = self.tokenizer([prompt]).to(device=device, non_blocking=True)
                    # Calculate the HPS
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(image, text)
                        image_features, text_features = outputs["image_features"], outputs["text_features"]
                        logits_per_image = image_features @ text_features.T
                        hps_score = torch.diagonal(logits_per_image).cpu().numpy()
                        max_hps_score = max(max_hps_score, hps_score[0])
                        #reward_hps_list.append(hps_score[0])
            #mean_hps_score = np.mean(reward_hps_list)
            result.append(max_hps_score)
        return result
