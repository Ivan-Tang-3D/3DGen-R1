"""
UnifiedReward-7B 奖励函数实现
基于vLLM API的多维度图像评估系统
支持 Alignment、Coherence、Style 三个维度的评分
"""

import torch
import numpy as np
from typing import List, Dict, Any
import re
from abc import ABC
from PIL import Image
import requests
import json
import base64
from io import BytesIO


class UnifiedReward3D:
    """基于UnifiedReward的3D多视角奖励评估"""
    
    def __init__(self, config=None):
        # 兼容 GRPOConfig 对象和字典
        if hasattr(config, '__dict__'):  # GRPOConfig对象
            self.api_base = getattr(config, 'unified_api_base', 'http://localhost:8090/v1')
            self.api_model_name = getattr(config, 'unified_api_model_name', 'UnifiedReward')
            self.num_views = getattr(config, 'unified_reward_num_views', 6)
            self.temperature = getattr(config, 'unified_reward_temperature', 0.0)
            self.aggregate_method = getattr(config, 'unified_aggregate_method', 'mean')  # 'mean' or 'min'
            self.score_weights = getattr(config, 'unified_score_weights', {'alignment': 1.0, 'coherence': 1.0, 'style': 1.0})
        else:  # 字典或None
            config_dict = config or {}
            self.api_base = config_dict.get('unified_api_base', 'http://localhost:8090/v1')
            self.api_model_name = config_dict.get('unified_api_model_name', 'UnifiedReward')
            self.num_views = config_dict.get('unified_reward_num_views', 6)
            self.temperature = config_dict.get('unified_reward_temperature', 0.0)
            self.aggregate_method = config_dict.get('unified_aggregate_method', 'mean')
            self.score_weights = config_dict.get('unified_score_weights', {'alignment': 1.0, 'coherence': 1.0, 'style': 1.0})
        
        # vLLM API endpoints
        self.chat_endpoint = f"{self.api_base}/chat/completions"
        self.accelerator = None
        self.device = None
        
        # UnifiedReward的评估prompt模板
        self.reward_prompt_template = """You are presented with a generated 3D model visualization and its associated text caption.
Your task is to analyze the visualization across multiple dimensions in relation to the caption. Specifically:

Provide overall assessments for the 3D model along the following axes (each rated from 1 to 5):
- Alignment Score: How well the 3D model matches the caption in terms of content and structure.
- Coherence Score: How logically consistent the 3D model is (absence of visual glitches, geometry distortions, etc.).
- Style Score: How aesthetically appealing the 3D model looks, regardless of caption accuracy.

Output your evaluation using the format below:

Alignment Score (1-5): X
Coherence Score (1-5): Y
Style Score (1-5): Z

Your task is provided as follows:
Text Caption: [{prompt}]"""

    @property
    def __name__(self):
        return 'UnifiedReward3D'
    
    def load_to_device(self, device):
        """设置设备信息 - vLLM API不需要本地加载模型"""
        self.device = device
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """将PIL图像转换为base64编码字符串"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    def _call_vllm_api(self, messages: List[Dict[str, Any]], max_tokens: int = 512) -> str:
        """调用vLLM API服务"""
        headers = {"Content-Type": "application/json"}
        
        data = {
            "model": self.api_model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "do_sample": False if self.temperature == 0.0 else True,
        }
        
        try:
            response = requests.post(
                self.chat_endpoint, 
                headers=headers, 
                json=data, 
                timeout=60,
                proxies={"http": None, "https": None}
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"❌ Error calling UnifiedReward API: {e}")
            return ""
        except (KeyError, IndexError) as e:
            print(f"❌ Error parsing UnifiedReward API response: {e}")
            return ""
    
    def _preprocess_views(self, views) -> List[Image.Image]:
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
                        print(f"⚠️  Invalid array shape at index {i}: {v.shape}")
                else:
                    print(f"⚠️  Unsupported view type at index {i}: {type(v)}")
            except Exception as e:
                print(f"⚠️  Failed to process view {i}: {e}")
                
        return view_images

    def compute_reward(self, data: Dict[str, Any]) -> torch.Tensor:
        """计算UnifiedReward分数（基于多视角图像）"""
        prompts = data.get('prompts', [])
        multi_view_images = data.get('multi_view_images', [])  # List[List[np.ndarray]]
        stage = data.get('stage', None)  # 获取当前阶段信息
        
        # 根据stage动态调整使用的维度
        if stage == 'stage1':
            # Stage 1 (high-level): 只使用alignment
            active_weights = {
                'alignment': 1.0,
                'coherence': 0.0,
                'style': 0.0
            }
            print(f"📊 UnifiedReward Stage 1: Using only Alignment")
        elif stage == 'stage2':
            # Stage 2 (low-level): 使用所有维度
            active_weights = {
                'alignment': 1.0,
                'coherence': 1.0,
                'style': 1.0
            }
            print(f"📊 UnifiedReward Stage 2: Using Alignment + Coherence + Style")
        else:
            # 默认使用所有维度
            active_weights = {
                'alignment': 1.0,
                'coherence': 1.0,
                'style': 1.0
            }
    
        scores = []
        for i, (prompt, views) in enumerate(zip(prompts, multi_view_images)):
            try:
                view_images = self._preprocess_views(views)

                if len(view_images) == 0:
                    # 如果没有有效的视角图像，返回0分
                    unified_score = 0.0
                else:
                    # 对每个视角进行评估
                    per_view_scores = []
                    for view_img in view_images:
                        alignment, coherence, style = self._compute_single_view_scores(prompt, view_img)
                        # 加权组合三个维度
                        view_score = (
                            alignment * self.score_weights.get('alignment', 1.0) +
                            coherence * self.score_weights.get('coherence', 1.0) +
                            style * self.score_weights.get('style', 1.0)
                        )
                        #     alignment * active_weights.get('alignment', 0.0) +
                        #     coherence * active_weights.get('coherence', 0.0) +
                        #     style * active_weights.get('style', 0.0)
                        # )
                        
                        per_view_scores.append(view_score)
                    
                    # 聚合多视角分数
                    if self.aggregate_method == 'mean':
                        unified_score = np.mean(per_view_scores)
                    elif self.aggregate_method == 'min':
                        unified_score = np.min(per_view_scores)
                    elif self.aggregate_method == 'max':
                        unified_score = np.max(per_view_scores)
                    else:
                        unified_score = np.mean(per_view_scores)
                    
                scores.append(unified_score)
                
            except Exception as e:
                print(f"❌ Error computing UnifiedReward for sample {i}: {e}")
                scores.append(0.0)
        
        return scores
    
    def _compute_single_view_scores(self, prompt: str, image: Image.Image) -> tuple:
        """计算单个视角的三个维度分数"""
        try:
            # 处理图像格式
            if not isinstance(image, Image.Image):
                if isinstance(image, np.ndarray):
                    # 处理numpy数组格式
                    if image.dtype != np.uint8:
                        image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                    image = Image.fromarray(image)
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")
                
            query_prompt = self.reward_prompt_template.format(prompt=prompt)
            
            # 将图像转换为base64
            image_base64 = self._image_to_base64(image)
            
            # 构建vLLM API格式的消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        {"type": "text", "text": query_prompt}
                    ]
                }
            ]
            
            # 调用vLLM API
            generated_text = self._call_vllm_api(messages, max_tokens=512)
            
            # 提取三个维度的分数
            alignment, coherence, style = self._extract_scores(generated_text)
            
            print(f"📊 UnifiedReward - Alignment: {alignment:.2f}, Coherence: {coherence:.2f}, Style: {style:.2f}")
            return alignment, coherence, style
            
        except Exception as e:
            print(f"❌ Error in UnifiedReward computation: {e}")
            return 0.0, 0.0, 0.0
    
    def _extract_scores(self, text: str) -> tuple:
        """从输出文本中提取三个维度的分数"""
        try:
            # 定义正则表达式模式
            alignment_pattern = r'Alignment Score \(1-5\):\s*([0-9.]+)'
            coherence_pattern = r'Coherence Score \(1-5\):\s*([0-9.]+)'
            style_pattern = r'Style Score \(1-5\):\s*([0-9.]+)'
            
            # 提取分数
            alignment_match = re.search(alignment_pattern, text)
            coherence_match = re.search(coherence_pattern, text)
            style_match = re.search(style_pattern, text)
            
            alignment = float(alignment_match.group(1)) if alignment_match else 0.0
            coherence = float(coherence_match.group(1)) if coherence_match else 0.0
            style = float(style_match.group(1)) if style_match else 0.0
            
            # 确保分数在1-5范围内
            alignment = max(1.0, min(5.0, alignment))
            coherence = max(1.0, min(5.0, coherence))
            style = max(1.0, min(5.0, style))
            
            return alignment, coherence, style
            
        except Exception as e:
            print(f"❌ Error extracting scores from text: {e}")
            print(f"Text was: {text}")
            return 0.0, 0.0, 0.0