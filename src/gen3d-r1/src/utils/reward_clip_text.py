"""
CLIP Text Reward for 3D Generation
基于CLIP的文本-图像对齐奖励函数，用于3D多视角评估
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers import CLIPModel, CLIPConfig
from safetensors.torch import load_file


class CLIPTextReward:
    """基于CLIP的文本-图像对齐奖励函数"""
    
    def __init__(self, config=None):
        """
        初始化CLIP Text Reward
        
        Args:
            config: 配置对象，可以是GRPOConfig或字典
        """
        # 兼容 GRPOConfig 对象和字典
        if hasattr(config, '__dict__'):  # GRPOConfig对象
            self.clip_model_path = getattr(config, 'clip_model_path', 'openai/clip-vit-large-patch14')
            self.num_views = getattr(config, 'clip_reward_num_views', 6)
            self.aggregate_method = getattr(config, 'clip_aggregate_method', 'max')  # 'mean', 'min', 'max'
            self.score_scale = getattr(config, 'clip_score_scale', 5.0)  # 分数缩放因子
        else:  # 字典或None
            config_dict = config or {}
            self.clip_model_path = config_dict.get('clip_model_path', 'openai/clip-vit-large-patch14')
            self.num_views = config_dict.get('clip_reward_num_views', 6)
            self.aggregate_method = config_dict.get('clip_aggregate_method', 'max')
            self.score_scale = config_dict.get('clip_score_scale', 5.0)
        
        # 模型组件（延迟加载）
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = None
        self.accelerator = None
        
    @property
    def __name__(self):
        return 'CLIPTextReward'
    
    def load_to_device(self, device):
        """加载CLIP模型到指定设备"""
        self.device = device
        model_path = '/mnt/petrelfs/tangyiwen/models/clip-vit-large-patch14'
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            num_cpus = os.cpu_count()
        
        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
        
        print(f'🔄 Loading CLIP model: {model_path}')
        
        try:
            # 先加载到 CPU，然后再移动到目标设备（避免分布式训练问题）
            self.model = CLIPModel.from_pretrained(
                model_path,
                torch_dtype=torch.float32,  # 使用 float32 而不是自动类型
                local_files_only=True
            )
            
            # # 验证模型权重
            # print(f"🔍 Verifying model weights...")
            # if hasattr(self.model.vision_model, 'embeddings'):
            #     patch_embed = self.model.vision_model.embeddings.patch_embedding
            #     print(f"   patch_embedding weight shape: {patch_embed.weight.shape}")
            #     if patch_embed.weight.ndim < 3:
            #         raise RuntimeError(f"Invalid patch_embedding weight shape: {patch_embed.weight.shape}")
            
            # # 现在移动到目标设备
            # self.model = self.model.to(device)
            # config = CLIPConfig.from_pretrained(model_path)
            # self.model = CLIPModel(config)
            # 加载safetensors格式的权重
            # safetensors_path = f"{model_path}/model.safetensors"
            # if os.path.exists(safetensors_path):
            #     state_dict = load_file(safetensors_path)
            # missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            print(f"🔍 Verifying model weights...")
            if hasattr(self.model.vision_model, 'embeddings'):
                patch_embed = self.model.vision_model.embeddings.patch_embedding
                print(f"   patch_embedding weight shape: {patch_embed.weight.shape}")
                if patch_embed.weight.ndim < 3:
                    raise RuntimeError(f"Invalid patch_embedding weight shape: {patch_embed.weight.shape}")
            
        except Exception as e:
            print(f"⚠️  Failed to load {self.clip_model_path}: {e}")
            print(f"🔄 Falling back to openai/clip-vit-base-patch32...")
            
            # 降级到 base 模型
            fallback_model = "openai/clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(
                fallback_model,
                torch_dtype=torch.float32,
            )
            self.model = self.model.to(device)
            self.clip_model_path = fallback_model
            print(f"✅ Loaded fallback model: {fallback_model}")
        
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 冻结模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        
        self.model.to(device)
        
        print(f'✅ CLIP model loaded to {device}')
    
    def _preprocess_views(self, views) -> List[Image.Image]:
        """预处理多视角图像"""
        view_images = []
        if not isinstance(views, list):
            return view_images
        
        for i, v in enumerate(views):
            try:
                if isinstance(v, Image.Image):
                    # 确保是RGB模式
                    if v.mode != 'RGB':
                        v = v.convert('RGB')
                    view_images.append(v)
                elif isinstance(v, np.ndarray):
                    # 处理numpy数组
                    if v.ndim == 2:
                        # 灰度图，转换为RGB
                        v = np.stack([v] * 3, axis=2)
                    elif v.ndim == 3:
                        # 检查通道数
                        if v.shape[2] == 1:
                            # 单通道，转换为RGB
                            v = np.repeat(v, 3, axis=2)
                        elif v.shape[2] == 4:
                            # RGBA，转换为RGB
                            v = v[:, :, :3]
                        elif v.shape[2] not in [3]:
                            print(f"⚠️  Invalid array shape at index {i}: {v.shape}")
                            continue
                    else:
                        print(f"⚠️  Invalid array dimensions at index {i}: {v.shape}")
                        continue
                    
                    # 确保是uint8类型
                    if v.dtype != np.uint8:
                        if v.max() <= 1.0:
                            v = (v * 255).astype(np.uint8)
                        else:
                            v = v.astype(np.uint8)
                    
                    view_images.append(Image.fromarray(v))
                    
                elif isinstance(v, torch.Tensor):
                    # 处理torch tensor
                    v_np = v.cpu().numpy()
                    
                    if v_np.ndim == 2:
                        # 灰度图
                        v_np = np.stack([v_np] * 3, axis=2)
                    elif v_np.ndim == 3:
                        # 如果是CHW格式，转换为HWC
                        if v_np.shape[0] in [1, 3, 4] and v_np.shape[0] < v_np.shape[1]:
                            v_np = np.transpose(v_np, (1, 2, 0))
                        
                        # 检查通道数
                        if v_np.shape[2] == 1:
                            v_np = np.repeat(v_np, 3, axis=2)
                        elif v_np.shape[2] == 4:
                            v_np = v_np[:, :, :3]
                        elif v_np.shape[2] not in [3]:
                            print(f"⚠️  Invalid tensor shape at index {i}: {v_np.shape}")
                            continue
                    else:
                        print(f"⚠️  Invalid tensor dimensions at index {i}: {v_np.shape}")
                        continue
                    
                    # 确保是uint8类型
                    if v_np.dtype != np.uint8:
                        if v_np.max() <= 1.0:
                            v_np = (v_np * 255).astype(np.uint8)
                        else:
                            v_np = v_np.astype(np.uint8)
                    
                    view_images.append(Image.fromarray(v_np))
                else:
                    print(f"⚠️  Unsupported view type at index {i}: {type(v)}")
            except Exception as e:
                print(f"⚠️  Failed to process view {i}: {e}")
                import traceback
                traceback.print_exc()
                
        return view_images
    
    def _process_image(self, img):
        """处理单张图像"""
        # 调试：打印输入类型和形状
        if isinstance(img, torch.Tensor):
            print(f"🔍 Debug: Input is torch.Tensor with shape {img.shape}")
        elif isinstance(img, np.ndarray):
            print(f"🔍 Debug: Input is np.ndarray with shape {img.shape}")
        elif isinstance(img, Image.Image):
            print(f"🔍 Debug: Input is PIL.Image with size {img.size} and mode {img.mode}")
        
        # 确保输入是PIL Image
        if isinstance(img, torch.Tensor):
            img = img.float()
            # 转换为PIL图像
            if img.ndim == 3:
                if img.shape[0] in [1, 3, 4]:  # CHW格式
                    img = img.permute(1, 2, 0)
                img_np = img.cpu().numpy()
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)
                # 确保是3通道
                if img_np.shape[2] == 1:
                    img_np = np.repeat(img_np, 3, axis=2)
                elif img_np.shape[2] == 4:
                    img_np = img_np[:, :, :3]
                img = Image.fromarray(img_np)
        elif isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            # 确保是3通道
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=2)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            # 确保是RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(img)}")
        
        print(f"🔍 Debug: After conversion, PIL.Image size={img.size}, mode={img.mode}")
        
        # 使用processor处理图像，确保返回正确的格式
        processed = self.processor(images=img, return_tensors="pt")
        
        if 'pixel_values' in processed:
            print(f"🔍 Debug: After processor, pixel_values shape = {processed['pixel_values'].shape}")
        
        return processed
    
    def _process_text(self, text):
        """处理文本"""
        return self.tokenizer(text, truncation=True, padding=True, return_tensors='pt')
    
    @torch.no_grad()
    def _calculate_clip_score(self, image_tensor, text_tensor):
        """计算CLIP分数"""
        if self.model is None:
            raise RuntimeError("CLIP model not loaded. Call load_to_device() first.")
        
        # 将输入移到正确的设备
        for key in image_tensor:
            image_tensor[key] = image_tensor[key].to(self.device)
        for key in text_tensor:
            text_tensor[key] = text_tensor[key].to(self.device)
        
        # 获取特征
        image_features = self.model.get_image_features(**image_tensor)
        text_features = self.model.get_text_features(**text_tensor)
        
        # 归一化特征
        image_features = image_features / image_features.norm(dim=1, keepdim=True).to(torch.float32)
        text_features = text_features / text_features.norm(dim=1, keepdim=True).to(torch.float32)
        
        # 计算余弦相似度
        score = (image_features * text_features).sum()
        return score.item() / image_features.shape[0]
    
    def compute_reward(self, data: Dict[str, Any]) -> List[float]:
        """
        计算CLIP奖励分数
        
        Args:
            data: 包含以下键的字典:
                - prompts: List[str] - 文本提示列表
                - multi_view_images: List[List[np.ndarray]] - 多视角图像列表
        
        Returns:
            List[float]: 每个样本的奖励分数
        """
        if self.model is None:
            raise RuntimeError("CLIP model not loaded. Call load_to_device() first.")
        
        prompts = data.get('prompts', [])
        multi_view_images = data.get('multi_view_images', [])
        
        scores = []
        for i, (prompt, views) in enumerate(zip(prompts, multi_view_images)):
            try:
                view_images = self._preprocess_views(views)
                
                if len(view_images) == 0:
                    # 如果没有有效的视角图像，返回0分
                    clip_score = 0.0
                else:
                    # 处理文本
                    text_tensor = self._process_text(prompt)
                    
                    # 对每个视角计算CLIP分数
                    per_view_scores = []
                    for view_img in view_images:
                        image_tensor = self._process_image(view_img)
                        view_score = self._calculate_clip_score(image_tensor, text_tensor)
                        per_view_scores.append(view_score)
                    
                    # 聚合多视角分数
                    if self.aggregate_method == 'mean':
                        clip_score = np.mean(per_view_scores)
                    elif self.aggregate_method == 'min':
                        clip_score = np.min(per_view_scores)
                    elif self.aggregate_method == 'max':
                        clip_score = np.max(per_view_scores)
                    else:
                        clip_score = np.mean(per_view_scores)
                    
                    # 缩放分数
                    clip_score = clip_score * self.score_scale
                
                scores.append(clip_score)
                
            except Exception as e:
                print(f"❌ Error computing CLIP score for sample {i}: {e}")
                import traceback
                traceback.print_exc()
                scores.append(0.0)
        
        return scores


# 向后兼容的别名
CLIPTextReward3D = CLIPTextReward
