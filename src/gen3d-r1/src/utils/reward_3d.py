"""
ReasonGen-R1风格的3D Reward实现
基于Qwen-2.5-VL的VLM reward评估系统
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
import re
from abc import ABC, abstractmethod
from PIL import Image
import trimesh
import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.models.sparse_structure_vqvae import VQVAE3D
import requests
import json
import base64
from io import BytesIO


class Base3DRewardModel(ABC):
    """3D奖励函数基类 遵循ReasonGen-R1的设计模式"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.device = None
        
    @abstractmethod
    def compute_reward(self, data: Dict[str, Any]) -> torch.Tensor:
        """计算奖励分数
        
        Args:
            data: 包含prompts, responses, mesh_tokens等信息的字典
            
        Returns:
            torch.Tensor: 奖励分数张量
        """
        pass
    
    def load_to_device(self, device):
        """加载到指定设备"""
        self.device = device


class VLM3DReward(Base3DRewardModel):
    """基于VLM的3D reward评估 - 使用vLLM API服务"""
    
    def __init__(self, config=None):
        super().__init__(config)
        # 兼容 GRPOConfig 对象和字典
        if hasattr(config, '__dict__'):  # GRPOConfig对象
            self.temperature = getattr(config, 'vlm_reward_temperature', 0.1)
            self.joint_multiview = getattr(config, 'vlm_reward_joint_multiview', False)
            self.strict_all_views = getattr(config, 'vlm_reward_strict_all_views', False)
            self.num_views = getattr(config, 'vlm_reward_num_views', 6)
            self.api_base = getattr(config, 'vlm_api_base', 'http://localhost:8000/v1')
            self.api_model_name = getattr(config, 'vlm_api_model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
        else:  # 字典或None
            config_dict = config or {}
            self.temperature = config_dict.get('vlm_reward_temperature', 0.1)
            self.joint_multiview = config_dict.get('vlm_reward_joint_multiview', False)
            self.strict_all_views = config_dict.get('vlm_reward_strict_all_views', False)
            self.num_views = config_dict.get('vlm_reward_num_views', 6)
            self.api_base = config_dict.get('vlm_api_base', 'http://localhost:8000/v1')
            self.api_model_name = config_dict.get('vlm_api_model_name', 'Qwen/Qwen2.5-VL-7B-Instruct')
        
        # vLLM API endpoints
        self.chat_endpoint = f"{self.api_base}/chat/completions"
        self.accelerator = None
        
        # ReasonGen-R1的reward prompt模板 - 单视角版本
        self.reward_prompt_template = """You are an expert 3D model evaluator. You will be given a text prompt describing a 3D object and a rendered view of a generated 3D model.

Text prompt: "{prompt}"

Please carefully examine the provided 3D model visualization and evaluate the following aspects:

1. **Geometric Accuracy**: Does the 3D model's shape, structure, and proportions match the prompt description?
2. **Spatial Relationships**: Are the spatial relationships between different parts of the object correct?
3. **Material and Surface Properties**: Do the materials, textures, and surface details match the prompt?
4. **Completeness**: Are all the key elements mentioned in the prompt present in the 3D model?
5. **3D Consistency**: Is the object geometrically sound and physically plausible?

Evaluation Criteria:
- The 3D model should be a faithful 3D representation of the prompt
- All major structural elements should be present and correctly positioned
- The object should look realistic and well-formed
- Materials and textures should be appropriate for the described object
- Consider that this is one view of a 3D object - think about 3D structure and depth

Be extremely strict and precise in your evaluation:
- Only give a perfect score if the 3D model is an excellent match to the prompt
- Consider both the overall shape and fine details
- Pay attention to 3D-specific aspects like depth, volume, and spatial relationships

Please conduct a thorough but balanced evaluation:
Check if all required elements from the prompt are present and correctly positioned\n
Evaluate whether the 3D model has good structural integrity and aligns with the prompt\n
Only if the pbject matches the prompt perfectly, respond with: \\boxed{{1}}.
Otherwise, respond with: \\boxed{{0}}.

Only one number should appear inside the box."""


    @property
    def __name__(self):
        return 'VLM3DReward'
    
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
        }
        
        try:
            response = requests.post(self.chat_endpoint, headers=headers, json=data, timeout=30,proxies={"http": None, "https": None})
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling vLLM API: {e}")
            return ""
        except (KeyError, IndexError) as e:
            print(f"Error parsing vLLM API response: {e}")
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
                        print(f"Warning: Invalid array shape at index {i}: {v.shape}")
                else:
                    print(f"Warning: Unsupported view type at index {i}: {type(v)}")
            except Exception as e:
                print(f"Warning: Failed to process view {i}: {e}")
                
        return view_images
    
    def compute_reward(self, data: Dict[str, Any]) -> torch.Tensor:
        """计算VLM-based 3D reward
        """
        stage = data.get('stage', 'stage2')  # 默认使用stage2的详细评估
        
        if stage == 'stage1':
            return self.compute_reward_detailed(data)
        else:  # stage2 or default
            return self.compute_reward_alignment(data)

    def compute_reward_detailed(self, data: Dict[str, Any]) -> torch.Tensor:
        """计算VLM-based 3D reward（基于多视角图像）"""
        prompts = data.get('prompts', [])
        multi_view_images = data.get('multi_view_images', [])  # List[List[np.ndarray]]
    
        scores = []
        for i, (prompt, views) in enumerate(zip(prompts, multi_view_images)):
            try:
                view_images = self._preprocess_views(views)

                if len(view_images) == 0:
                    reward_score = 0.0
                elif self.joint_multiview:
                    # 将多视角一次性输入，得到单一boxed答案
                    reward_score = float(self._compute_joint_multiview_reward_detailed(prompt, view_images))
                else:
                    # 逐视角评估，然后聚合
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
        
        return scores

    def compute_reward_alignment(self, data: Dict[str, Any]) -> torch.Tensor:
        """计算VLM-based 3D reward（基于多视角图像）"""
        prompts = data.get('prompts', [])
        multi_view_images = data.get('multi_view_images', [])  # List[List[np.ndarray]]
    
        scores = []
        for i, (prompt, views) in enumerate(zip(prompts, multi_view_images)):
            try:
                view_images = self._preprocess_views(views)

                if len(view_images) == 0:
                    reward_score = 0.0
                elif self.joint_multiview:
                    # 将多视角一次性输入，得到单一boxed答案
                    reward_score = float(self._compute_joint_multiview_reward_alignment(prompt, view_images))
                else:
                    # 逐视角评估，然后聚合
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
        
        return scores
    
    def _compute_single_reward(self, prompt: str, image: Image.Image) -> float:
        """计算单个prompt-image对的reward - 使用vLLM API"""
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
                        {"type": "text", "text": query_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                }
            ]
            
            # 调用vLLM API
            generated_text = self._call_vllm_api(messages, max_tokens=512)
            
            # 提取boxed答案
            reward = self._extract_boxed_answer(generated_text)
            print(f"reward: {reward}")
            return float(reward)
            
        except Exception as e:
            print(f"Error in VLM reward computation: {e}")
            return 0.0

    def _compute_joint_multiview_reward_detailed(self, prompt: str, images: list[Image.Image]) -> float:
        """多视角合并评估：一次性输入多张视角图，返回归一化分数(0-1) - 使用vLLM API"""
        try:
        
            query_prompt = (
                "You are an expert 3D model evaluator. You are given a text prompt used to generate a 3D model: \"{prompt}\"\n\n"
                "Below are {num_views} different rendered views of the generated 3D object from different angles.\n\n"
                "Please evaluate the 3D model by examining all views comprehensively:\n\n"
                "Step 1: Describe the 3D object thoroughly\n"
                "Examine all rendered views and describe what you see as a complete 3D object. Focus on the overall shape, key structural features, materials, colors, and multi-view consistency. Do not be affected by the prompt in this description.\n\n"
                "Step 2: Score the 3D generation quality from the following aspects:\n"
                "- Does the 3D object match the category specified in the prompt when viewed from all angles? (0-1 score)\n"
                "- Is the 3D model structurally complete without missing parts across all views? (0-1 score)\n"
                "- Does the 3D object look realistic and geometrically reasonable with proper proportions? (0-1 score)\n"
                "- Are the rendered views clear, well-lit, and show good visual quality? (0-1 score)\n"
                "- Is the 3D model free of artifacts, inconsistencies between views, and structural defects? (0-1 score)\n\n"
                "Your response should be in a **JSON** format following rules below:\n"
                "1. The final response should have three keys: \"description\", \"score\", \"explanation\".\n"
                "2. The total score should be stored in the \"score\" key. You may use float numbers. The total score should be between 0 and 5.\n"
                "3. The reasoning and scoring process should be written in \"explanation\" to justify your score, considering all rendered views.\n\n"
                "Following is an example of response:\n"
                "```json\n"
                "{{\n"
                "\"description\": \"The 3D model shows a four-legged animal resembling a dog. From the front view, it has two visible ears and a snout. The side views reveal a complete body structure with four legs and a tail. The model appears to be textured with brown coloring. However, examining all views reveals that one eye is missing from the facial structure.\",\n"
                "\"score\": 3.5,\n"
                "\"explanation\": \"The 3D object matches the category 'a dog' as specified in the prompt when viewed from multiple angles (1 score). The model is structurally complete with all major body parts visible including head, body, four legs, and tail (1 score). However, the model looks somewhat unrealistic because it has only one eye visible in the front view, and the eye position is not anatomically reasonable (0 score). The rendered views are generally clear and well-lit, though some views show slight blurriness in the background (0.5 score). The model is free of major rendering artifacts and maintains reasonable consistency across views (1 score). The total score is 1+1+0+0.5+1=3.5.\"\n"
                "}}\n"
                "```"
            ).format(prompt=prompt, num_views=len(images))

            # 构建消息内容
            contents = [{"type": "text", "text": query_prompt}]
            
            # 将每个图像转换为base64并添加到内容中
            for img in images:
                if not isinstance(img, Image.Image):
                    if isinstance(img, np.ndarray):
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                        img = Image.fromarray(img)
                
                image_base64 = self._image_to_base64(img)
                contents.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                })

            messages = [{"role": "user", "content": contents}]
            
            # 调用vLLM API
            generated_text = self._call_vllm_api(messages, max_tokens=512)
            
            return self._extract_boxed_answer_detailed(generated_text)
        except Exception as e:
            print(f"Error in joint multiview reward computation: {e}")
            return 0.0
    
    def _extract_boxed_answer_detailed(self, text: str) -> float:
        """从JSON格式响应中提取分数并除以4
        
        期望的JSON格式:
        {
            "description": "...",
            "score": X.X,
            "explanation": "..."
        }
        
        Returns:
            float: 归一化的分数 (0-1之间，原始分数/4)
        """
        try:
            # 首先尝试从JSON格式中提取
            # 查找JSON代码块
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_matches = re.findall(json_pattern, text, re.DOTALL)
            
            if json_matches:
                json_str = json_matches[-1]  # 取最后一个匹配的JSON
                try:
                    data = json.loads(json_str)
                    if "score" in data:
                        raw_score = float(data["score"])
                        normalized_score = raw_score / 4.0  # 归一化到0-1
                        #normalized_score = raw_score
                        print(f"📊 Extracted score from JSON: {raw_score} -> normalized: {normalized_score:.4f}")
                        return normalized_score
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON parsing error: {e}")
            
            # 如果没有JSON代码块，尝试直接解析JSON
            try:
                # 清理文本，查找可能的JSON对象
                json_obj_pattern = r'\{[^{}]*"score"[^{}]*\}'
                json_obj_matches = re.findall(json_obj_pattern, text, re.DOTALL)
                
                for json_str in json_obj_matches:
                    try:
                        data = json.loads(json_str)
                        if "score" in data:
                            raw_score = float(data["score"])
                            normalized_score = raw_score / 4.0
                            #normalized_score = raw_score
                            print(f"📊 Extracted score from inline JSON: {raw_score} -> normalized: {normalized_score:.4f}")
                            return normalized_score
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                print(f"⚠️  Error parsing inline JSON: {e}")
            
            # Fallback 1: 查找 "score": 数字 的模式
            score_pattern = r'"score"\s*:\s*([0-9]+\.?[0-9]*)'
            score_matches = re.findall(score_pattern, text)
            
            if score_matches:
                raw_score = float(score_matches[-1])
                normalized_score = raw_score / 4.0
                #normalized_score = raw_score 
                print(f"📊 Extracted score from pattern: {raw_score} -> normalized: {normalized_score:.4f}")
                return normalized_score
            
            # Fallback 2: 查找任何浮点数（0-5范围）
            float_pattern = r'\b([0-5]\.?[0-9]*)\b'
            float_matches = re.findall(float_pattern, text)
            
            if float_matches:
                # 取最后一个看起来合理的分数
                for match in reversed(float_matches):
                    try:
                        raw_score = float(match)
                        if 0 <= raw_score <= 5:
                            normalized_score = raw_score / 4.0
                            #normalized_score = raw_score
                            print(f"⚠️  Using fallback score: {raw_score} -> normalized: {normalized_score:.4f}")
                            return normalized_score
                    except ValueError:
                        continue
            
            print("❌ No valid score found, returning 0")
            return 0.0
            
        except Exception as e:
            print(f"Error extracting score: {e}")
            return 0.0
        
    def _compute_joint_multiview_reward_alignment(self, prompt: str, images: list[Image.Image]) -> int:
        """多视角合并评估：一次性输入多张视角图，返回0/1 - 使用vLLM API"""
        try:
            # 为多视角3D评估构造专门的模板
            query_prompt = (
                "You are an expert 3D model evaluator. You will evaluate a generated 3D object against a text prompt using multiple rendered views.\n\n"
                "Text prompt: \"{prompt}\"\n\n"
                "Below are {num_views} different rendered views of the generated 3D object. These views show the object from different angles to give you a complete understanding of its 3D structure.\n\n"
                "Please follow these steps to evaluate the 3D model (limit each step to one paragraph, no bullet points or numbered sub-items):\n\n"
                "Step 1: Describe the 3D object independently (one paragraph)\n"
                "Examine all {num_views} views and describe what you see as a complete 3D object without being influenced by the prompt. Focus on the overall 3D shape, key features, materials, and multi-view consistency.\n\n"
                "Step 2: Extract key elements from the prompt (one paragraph)\n"
                "Analyze the text prompt and identify the main 3D elements and requirements, including object type, geometric features, materials, and any specific characteristics mentioned.\n\n"
                "Step 3: Evaluate (one paragraph)\n"
                "Assess whether the 3D model matches the prompt requirements across all views, considering element presence, positioning, structural integrity, and overall alignment. Provide your evaluation in a single flowing paragraph without using lists or bullet points.\n\n"
                "Please conduct a thorough but balanced evaluation with concise and clear responses:\n"
                "- Check if all required elements from the prompt are present and correctly positioned\n"
                "- Evaluate whether the 3D model has good structural integrity and aligns with the prompt\n"
                "- Only if the object matches the prompt perfectly across all views, respond with: \\boxed{{1}}.\n"
                "- Otherwise, respond with: \\boxed{{0}}.\n\n"
                "Only one number should appear inside the box."
            ).format(prompt=prompt, num_views=len(images))
            
            # 构建消息内容
            contents = [{"type": "text", "text": query_prompt}]
            
            # 将每个图像转换为base64并添加到内容中
            for img in images:
                if not isinstance(img, Image.Image):
                    if isinstance(img, np.ndarray):
                        if img.dtype != np.uint8:
                            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
                        img = Image.fromarray(img)
                
                image_base64 = self._image_to_base64(img)
                contents.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                })

            messages = [{"role": "user", "content": contents}]
            
            # 调用vLLM API
            generated_text = self._call_vllm_api(messages, max_tokens=512)
            
            return self._extract_boxed_answer_alignment(generated_text)
        except Exception as e:
            print(f"Error in joint multiview reward computation: {e}")
            return 0
    
    def _extract_boxed_answer_alignment(self, text: str) -> int:
        """提取\\boxed{x}中的答案"""
        try:
            # 查找\\boxed{数字}模式
            pattern = r'\\boxed\{(\d+)\}'
            matches = re.findall(pattern, text)
            
            if matches:
                return int(matches[-1])  # 取最后一个匹配
            else:
                print("⚠️  No \\boxed{} format found, trying fallback...")
                # 如果没有找到boxed格式，尝试查找数字
                numbers = re.findall(r'\b[01]\b', text)
                if numbers:
                    return int(numbers[-1])
                else:
                    print("❌ No valid 0/1 answers found, returning 0")
                    return 0
        except Exception as e:
            print(f"Error extracting boxed answer: {e}")
            return 0