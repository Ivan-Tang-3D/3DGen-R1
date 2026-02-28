"""
3D GRPO Trainer for T2I-R1
"""

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import numpy as np
import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
import imageio
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from transformers import Qwen2_5_VLForConditionalGeneration
import ipdb
import shutil
import atexit
from pathlib import Path
import glob

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

# 3D相关导入
import trimesh
import open3d as o3d
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
from trellis.models.sparse_structure_vqvae import VQVAE3D
import uuid
import tempfile
import copy
import re
import json
import random
from typing import List, Dict
from PIL import Image
from datetime import datetime
import pickle

# 导入3D reward functions 
from utils.reward_3d import VLM3DReward
from utils.reward_hps import HPSv2
from utils.reward_unified_3d import UnifiedReward3D
from utils.reward_clip_text import CLIPTextReward

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

RewardFunc3D = Union[str, PreTrainedModel, Callable[[list, list, list], list[float]]]


class JanusT2IR1Trainer3D(Trainer):

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc3D, list[RewardFunc3D]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
        script_args = None,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO-3D")

        self.mesh_token_num_per_model = getattr(args, 'mesh_token_num_per_model', 1024)
        self.vqvae_model_path = getattr(args, 'vqvae_model_path', None)
        self.trellis_model_path = getattr(args, 'trellis_model_path', "JeffreyXiang/TRELLIS-text-xlarge")
        self.simplify = getattr(args, 'simplify', 0.95)
        self.texture_size = getattr(args, 'texture_size', 1024)
        self.vlm_reward_num_views = getattr(args, 'vlm_reward_num_views', 6)
        self.video_save_interval = getattr(args, 'video_save_interval', 10) 

        self.tmp_dir = "./src/t2i-r1/src/tmp/Reasoning-ShapeLLM-vllm"
        os.makedirs(self.tmp_dir, exist_ok=True)
        # Models
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, trust_remote_code=True, torch_dtype=torch.bfloat16
            )
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )


        frozen_components = [
            "visual", "vision_tower", "vision_model", 
            "mm_projector", "multi_modal_projector", 
            "vision_resampler", "aligner","lm_head"
        ]
        for name, param in model.named_parameters():
            if any(component in name for component in frozen_components):
                param.requires_grad = False
           

        model.config.use_cache = False
        model.gradient_checkpointing_enable()
            
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled() and (args.beta != 0 or getattr(args, 'use_gspo', False)):
            self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, trust_remote_code=True
            )
        elif peft_config is None and (args.beta != 0 or getattr(args, 'use_gspo', False)):
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str) and 'vlm_3d' in reward_func:
                reward_funcs[i] = VLM3DReward(args)
            elif isinstance(reward_func, str) and 'hps' in reward_func:
                reward_funcs[i] = HPSv2(args)
            elif isinstance(reward_func, str) and 'unified' in reward_func:
                reward_funcs[i] = UnifiedReward3D(args)
            else:
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")
            
        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.new_generations_3d = getattr(args, 'new_generations_3d', 1)
        self.beta = args.beta


        self.mesh_start_token = "<mesh-start>"
        self.mesh_end_token = "<mesh-end>"
        self.mesh_token_pattern = r'<mesh(\d+)>'
        
        self.epsilon_low = getattr(args, 'epsilon_low', 0.2)
        self.epsilon_high = getattr(args, 'epsilon_high', 0.28)

        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.beta != 0 or self.use_gspo:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        else:
            self.ref_model = None

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
            elif isinstance(reward_func, VLM3DReward):
                reward_func.accelerator = self.accelerator
                reward_func.device = self.accelerator.device
                reward_func.load_to_device(self.accelerator.device)  # 只是设置设备信息
                print(f"✅ VLM3DReward configured to use vLLM API service")
               
            elif isinstance(reward_func, UnifiedReward3D):
                reward_func.accelerator = self.accelerator
                reward_func.device = self.accelerator.device
                reward_func.load_to_device(self.accelerator.device)  # 只是设置设备信息
                print(f"✅ UnifiedReward3D configured to use vLLM API service")
            elif isinstance(reward_func, HPSv2):
                reward_func.load_to_device(self.accelerator.device)
            elif hasattr(reward_func, 'load_to_device'):
                reward_func.load_to_device(self.accelerator.device)

        self._init_3d_components()
        
        # Two-stage CoT configuration
        self.enable_two_stage_cot = getattr(args, 'enable_two_stage_cot', False)  
        
        # reasoning prompt
        if hasattr(args, 'reasoning_prompt_path') and args.reasoning_prompt_path:
            with open(args.reasoning_prompt_path, 'r') as f:
                self.cot_prompt = f.read()
        else:
            # 3D reasoning prompt
            self.cot_prompt = """Let me think step by step about generating the 3D model for '{}'.

I need to consider:
1. Overall geometric structure: What is the basic shape and form of this object?
2. Key features and details: What distinctive elements define this object?  
3. Proportions and scale: How do different parts relate in size and position?
4. Material properties: What surface textures and materials are needed?
5. Functional aspects: How does form follow function for this object?
6. Spatial relationships: How do components connect and interact in 3D space?
7. Topology considerations: What is the surface topology and mesh structure?
8. Rendering properties: How should this object appear when rendered?

My detailed reasoning process:"""

        
    def freeze_trellis_pipeline_comprehensive(self):
        if self.pipeline_text is None:
            print("❌ Trellis pipeline not loaded")
            return
        
        total_frozen = 0
        
        if hasattr(self.pipeline_text, 'models') and self.pipeline_text.models:
            print("📋 Check main models (self.models):")
            for model_name, model in self.pipeline_text.models.items():
                if hasattr(model, 'parameters'):
                    param_count = sum(p.numel() for p in model.parameters())
                    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    
                    for param in model.parameters():
                        param.requires_grad = False
                    model.eval()
                    
                    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    frozen_count = trainable_before - trainable_after
                    total_frozen += frozen_count
                    
                    print(f"  🔒 {model_name}: {frozen_count:,} parameter frozen (total: {param_count:,})")
                else:
                    print(f"  ⚠️  {model_name}: no parameters() method")
        
        if hasattr(self.pipeline_text, 'text_cond_model') and self.pipeline_text.text_cond_model:
            print("📋  Check text condition model (self.text_cond_model):")
            if 'model' in self.pipeline_text.text_cond_model:
                clip_model = self.pipeline_text.text_cond_model['model']
                if hasattr(clip_model, 'parameters'):
                    param_count = sum(p.numel() for p in clip_model.parameters())
                    trainable_before = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
                    
                    for param in clip_model.parameters():
                        param.requires_grad = False
                    clip_model.eval()
                    
                    trainable_after = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
                    frozen_count = trainable_before - trainable_after
                    total_frozen += frozen_count
                    
                    print(f"  🔒 CLIP text model: {frozen_count:,} parameter frozen (total: {param_count:,})")
            
            for key, value in self.pipeline_text.text_cond_model.items():
                if key != 'model' and hasattr(value, 'parameters'):
                    print(f"  ⚠️  Extra Trainable Modules: {key}")
                    for param in value.parameters():
                        param.requires_grad = False
                        total_frozen += param.numel()
        
        samplers_to_check = [
            ('sparse_structure_sampler', getattr(self.pipeline_text, 'sparse_structure_sampler', None)),
            ('slat_sampler', getattr(self.pipeline_text, 'slat_sampler', None))
        ]
        
        print("📋 Check Sampler:")
        for sampler_name, sampler in samplers_to_check:
            if sampler is not None:
                if hasattr(sampler, 'parameters'):
                    param_count = sum(p.numel() for p in sampler.parameters())
                    if param_count > 0:
                        print(f"  ⚠️  {sampler_name} has {param_count:,} parameters!")
                        for param in sampler.parameters():
                            param.requires_grad = False
                            total_frozen += param.numel()
                    else:
                        print(f"  ✅ {sampler_name}: no parameters")
                else:
                    print(f"  ✅ {sampler_name}: no parameters() method")
            else:
                print(f"  ⚠️  {sampler_name}: not found")
        
        if hasattr(self.pipeline_text, 'parameters'):
            print("📋 Check pipeline itself:")
            try:
                pipeline_params = list(self.pipeline_text.parameters())
                if pipeline_params:
                    print(f"  ⚠️  Pipeline itself has {len(pipeline_params)} parameters!")
                    for param in pipeline_params:
                        param.requires_grad = False
                        total_frozen += param.numel()
                else:
                    print("  ✅ Pipeline itself has no parameters")    
            except Exception as e:
                print(f"  ✅ Pipeline itself has no parameters() method: {e}")
        
        print("📋 Deep scan all attributes:")
        checked_attrs = {'models', 'text_cond_model', 'sparse_structure_sampler', 'slat_sampler'}
        
        for attr_name in dir(self.pipeline_text):
            if not attr_name.startswith('_') and attr_name not in checked_attrs:
                attr_value = getattr(self.pipeline_text, attr_name, None)
                if attr_value is not None and hasattr(attr_value, 'parameters'):
                    try:
                        param_list = list(attr_value.parameters())
                        if param_list:
                            param_count = sum(p.numel() for p in param_list)
                            print(f"  ⚠️  Found unexamined model attribute {attr_name}: {param_count:,} parameters")
                            for param in param_list:
                                param.requires_grad = False
                                total_frozen += param.numel()
                    except:
                        pass
        
        print("="*60)
        print(f"✅ Trellis parameter frozen completed!")
        print(f"📊 Total {total_frozen:,} parameters frozen")
        print("="*60)
        
        return total_frozen
    
    def _init_3d_components(self):
        print("Initializing 3D components...")

        if self.vqvae_model_path:
            self.vqvae = VQVAE3D(num_embeddings=8192)
            state_dict = torch.load(self.vqvae_model_path, map_location="cpu")
            self.vqvae.load_state_dict(state_dict)
            self.vqvae.to(self.accelerator.device)

            for param in self.vqvae.parameters():
                param.requires_grad = False
            self.vqvae.eval()  
        else:
            self.vqvae = None
            print("Warning: VQVAE model path not provided, 3D generation will be limited")
        
        try:
            self.pipeline_text = TrellisTextTo3DPipeline.from_pretrained("./models/TRELLIS-text-xlarge")
            self.pipeline_text.to(self.accelerator.device)
    
            self.freeze_trellis_pipeline_comprehensive()
        except Exception as e:
            print(f"Warning: Failed to load Trellis pipeline: {e}")
            self.pipeline_text = None
    
        print("3D components initialized!.")
    
    def _transform_messages(self, original_messages):
        transformed_messages = []
        for message in original_messages:
            new_content = []
            for item in message['content']:
                if 'image' in item:
                    new_item = {'type': 'image', 'image': item['image']}
                elif 'text' in item:
                    new_item = {'type': 'text', 'text': item['text']}
                elif 'video' in item:
                    new_item = {'type': 'video', 'video': item['video']}
                else:
                    continue
                new_content.append(new_item)
            new_message = {'role': message['role'], 'content': new_content}
            transformed_messages.append(new_message)
        return transformed_messages

    def _create_detailed_trial_id(self, prompt_id, training_step, generation_idx):

        cleaned_prompt = re.sub(r'[^\w\s-]', '', str(prompt_id))
        cleaned_prompt = re.sub(r'[-\s]+', '_', cleaned_prompt)
        if len(cleaned_prompt) > 50:
            cleaned_prompt = cleaned_prompt[:50]
        
        trial_id = f"step_{training_step:06d}_gen_{generation_idx:03d}_{cleaned_prompt}"
        return trial_id

    def _save_video_safely(self, video, trial_id):
        video_path = f"{self.tmp_dir}/{trial_id}.mp4"
        
        try:
            imageio.mimsave(video_path, video, fps=15, codec='libx264')
            print(f"✅ Video saved: {video_path}")
            return video_path
        except Exception as mp4_error:
            print(f"⚠️  MP4 encoding failed: {mp4_error}")
            try:
                gif_path = f"{self.tmp_dir}/{trial_id}.gif"
                imageio.mimsave(gif_path, video, fps=10)
                print(f"✅ Video saved as GIF: {gif_path}")
                return gif_path
            except Exception as gif_error:
                print(f"⚠️  GIF encoding also failed: {gif_error}")
                try:
                    avi_path = f"{self.tmp_dir}/{trial_id}.avi"
                    imageio.mimsave(avi_path, video, fps=15, codec='rawvideo')
                    print(f"✅ Video saved as AVI: {avi_path}")
                    return avi_path
                except Exception as avi_error:
                    print(f"❌ All video encoding methods failed: {avi_error}")

                    frames_dir = f"{self.tmp_dir}/{trial_id}_frames"
                    os.makedirs(frames_dir, exist_ok=True)
                    for i, frame in enumerate(video):
                        frame_path = f"{frames_dir}/frame_{i:04d}.png"
                        imageio.imwrite(frame_path, frame)
                    print(f"✅ Video frames saved to: {frames_dir}")
                    return frames_dir
                
    def _get_per_token_logps(self, model, input_embeds, text_ids, mesh_ids, attention_mask):
        def _get_per_token_logps_part(logits, input_ids):
            logits = logits[:, :-1, :]
            input_ids = input_ids[:, 1:]
            per_token_logps = []

            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        if mesh_ids is not None:
            hidden_states = model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states
            last_hidden_states = hidden_states[-1]

            if hasattr(model, 'mesh_head'):
                mesh_logits = model.mesh_head(last_hidden_states[:, -(mesh_ids.size(1)+1):, :])
            else:
                mesh_logits = model.lm_head(last_hidden_states[:, -(mesh_ids.size(1)+1):, :])
            
            mesh_input_ids = torch.cat([mesh_ids.new_zeros(mesh_ids.size(0), 1), mesh_ids], dim=1)
            per_token_logps_mesh = _get_per_token_logps_part(mesh_logits, mesh_input_ids)
            return torch.cat([
                per_token_logps_mesh.new_zeros(
                    (per_token_logps_mesh.size(0), input_embeds.size(1) - per_token_logps_mesh.size(1) - 1)
                ),
                per_token_logps_mesh
            ], dim=1)
        else:
            hidden_states = model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states
            last_hidden_states = hidden_states[-1]
            text_logits = model.lm_head(last_hidden_states)
            per_token_logps_text = _get_per_token_logps_part(text_logits, text_ids)
            return per_token_logps_text

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def _parse_mesh_tokens(self, response_text):
        """解析生成的mesh tokens"""
        import re
        try:
            matches = re.findall(self.mesh_token_pattern, response_text)
            mesh_tokens = []
            
            if matches:
                print(f"Found {len(matches)} mesh tokens using regex")
                for match in matches:
                    try:
                        mesh_tokens.append(int(match))
                    except ValueError:
                        continue
            
            if len(mesh_tokens) == 0:
                mesh_tokens = [0] * self.mesh_token_num_per_model
            elif len(mesh_tokens) < self.mesh_token_num_per_model:

                fill_value = mesh_tokens[-1] if mesh_tokens else 0
                while len(mesh_tokens) < self.mesh_token_num_per_model:
                    mesh_tokens.append(fill_value)
            elif len(mesh_tokens) > self.mesh_token_num_per_model:
                mesh_tokens = mesh_tokens[:self.mesh_token_num_per_model]
            
            return torch.tensor(mesh_tokens, dtype=torch.long).unsqueeze(0)
            
        except Exception as e:
            print(f"Error parsing mesh tokens: {e}")
            return torch.zeros(1, self.mesh_token_num_per_model, dtype=torch.long)

    def _generate_3d_model(self, mesh_tokens, enhanced_prompt, prompt_id="", training_step=0, generation_idx=0):
        """从mesh tokens生成3D模型"""
        
        if self.vqvae is None or self.pipeline_text is None:
            print("⚠️  3D generation components not loaded")
            return None, None
        
        should_save_video = (training_step % self.video_save_interval == 0) or training_step == 0
        
        target_device = self.accelerator.device
        original_device = mesh_tokens.device
        video, video_org = None, None
        
        try:
            recon = self.vqvae.Decode(mesh_tokens.to(target_device))
            
            z_s = recon[0].detach() #.cpu()
            z_s = (z_s > 0) * 1
            indices = torch.nonzero(z_s[0] == 1)
            position_recon = (indices.float() + 0.5) / 64 - 0.5
            coords = ((position_recon + 0.5) * 64).int().contiguous()
            ss = torch.zeros(1, 64, 64, 64, dtype=torch.long, device=target_device)
            ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
            ss = ss.unsqueeze(0)
            coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

            with torch.no_grad():
                cond = self.pipeline_text.get_cond([enhanced_prompt])
                slat = self.pipeline_text.sample_slat(cond, coords.to(target_device))
                outputs = self.pipeline_text.decode_slat(slat, ['mesh', 'gaussian'])

            try:
                video_org = render_utils.render_video(outputs['gaussian'][0], num_frames=60)['color']
                video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=60)['normal']
                video = [np.concatenate([video_org[i], video_geo[i]], axis=1) for i in range(len(video_org))]
            except Exception as e:
                print(f"Video rendering failed: {e}")
                video = None
                video_org = None
            
            if should_save_video and video is not None:
                trial_id = self._create_detailed_trial_id(prompt_id, training_step, generation_idx)
                video_path = self._save_video_safely(video, trial_id)
                print(f"✅ Video saved at step {training_step} (interval: {self.video_save_interval})")
                
            return video, video_org
            
        except Exception as e:
            print(f"3D model generation failed: {e}")
            return None, None
        
    def _generate_cot_stage(self, model, prompts, cot_template, max_tokens, num_gens=None):
        """Helper to generate CoT for one stage"""
        if num_gens is None:
            num_gens = self.num_generations
        
        # Build reasoning prompts
        reasoning_prompts = [cot_template.format(p) for p in prompts]
        
        # Format and tokenize
        formatted_texts = []
        system_prompt = "You are a helpful assistant that receives a 3D prompt and generate a visualization of the prompt."
        for reasoning_prompt in reasoning_prompts:
            messages = [
                {'role': 'system', 'content': [{'text': system_prompt}]},
                {'role': 'user', 'content': [{'text': reasoning_prompt}]}
            ]
            messages = self._transform_messages(messages)
            text = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_texts.append(text)
        
        prompt_inputs = self.processing_class(text=formatted_texts, padding=True, return_tensors='pt')
        prompt_ids = prompt_inputs["input_ids"].to(self.accelerator.device)
        prompt_mask = prompt_inputs["attention_mask"].to(self.accelerator.device)
        
        
        # Generate CoT
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.config.use_cache = False
            unwrapped_model.gradient_checkpointing_disable()
            
            prompt_ids_expanded = prompt_ids.repeat_interleave(num_gens, dim=0)
            prompt_mask_expanded = prompt_mask.repeat_interleave(num_gens, dim=0)
            input_embeds = unwrapped_model.get_input_embeddings()(prompt_ids_expanded)
            
            completion_ids = unwrapped_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=prompt_mask_expanded,
                max_new_tokens=max_tokens,
                pad_token_id=self.processing_class.tokenizer.eos_token_id,
                eos_token_id=self.processing_class.tokenizer.eos_token_id,
                bos_token_id=self.processing_class.tokenizer.bos_token_id,
                do_sample=True,
                use_cache=True
            )
        
        return prompt_ids, prompt_mask, completion_ids

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer3D does not support returning outputs")
        
        prompts = [x["raw_prompt"] for x in inputs]
        
        return self._compute_loss_single_stage(model, inputs)
        

    def _compute_loss_single_stage(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer3D does not support returning outputs")
        
        prompts = [x["raw_prompt"] for x in inputs]
        
        reasoning_prompts = [
            self.cot_prompt.format(prompt[0]["content"] if isinstance(prompt, list) else prompt)
            for prompt in prompts
        ]
        
        formatted_texts = []
        system_prompt = "You are a helpful assistant that receives a 3D prompt and generate a visualization of the prompt."
        
        for reasoning_prompt in reasoning_prompts:
            messages = [
                {'role': 'system', 'content': [{'text': system_prompt}]},
                {'role': 'user', 'content': [{'text': reasoning_prompt}]}
            ]
            messages = self._transform_messages(messages)
            text = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            formatted_texts.append(text)

        prompt_inputs = self.processing_class(text=formatted_texts, images=None, videos=None, 
                               padding=True, return_tensors='pt')
        
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        
        device = self.accelerator.device
        prompt_ids = prompt_ids.to(device)
        prompt_mask = prompt_mask.to(device)
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        
        torch.set_grad_enabled(False)
        
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.config.use_cache = False
            unwrapped_model.gradient_checkpointing_disable()
            
            prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            input_embeds = unwrapped_model.get_input_embeddings()(prompt_ids)
            
            prompt_completion_ids = unwrapped_model.generate(
                inputs_embeds=input_embeds,
                attention_mask=prompt_mask,
                pad_token_id=self.processing_class.tokenizer.eos_token_id,
                bos_token_id=self.processing_class.tokenizer.bos_token_id,
                eos_token_id=self.processing_class.tokenizer.eos_token_id,
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                use_cache=True
            )
            
            prompt_length = prompt_ids.size(1)
            
            if self.max_completion_length is not None:
                prompt_completion_ids = prompt_completion_ids[:, -self.max_completion_length:]
            
            completion_ids = prompt_completion_ids
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.tokenizer.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        loss_dict = {}
        model.module.gradient_checkpointing_enable()
        torch.set_grad_enabled(True)
        
        prompt_all_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        input_embeds = model.module.get_input_embeddings()(prompt_all_ids)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        per_token_logps = self._get_per_token_logps(
            model=model.module,
            input_embeds=input_embeds,
            text_ids=prompt_all_ids,
            mesh_ids=None,
            attention_mask=attention_mask
        )
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        
        with torch.inference_mode():
            if self.ref_model is not None:
                self.ref_model.gradient_checkpointing_enable()
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    input_embeds,
                    prompt_all_ids,
                    None,
                    attention_mask
                )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            else:
                ref_per_token_logps = torch.zeros_like(per_token_logps)
        
        loss_dict['semantic-cot'] = {
            'per_token_logps': per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
            'completion_mask': completion_mask,
        }
        torch.set_grad_enabled(False)
        
        gen_prompt_list = []
        enhanced_prompt_list = []
        for i in range(completion_ids.shape[0]):
            reasoning_text = self.processing_class.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True)
            raw_prompt = inputs[i // self.num_generations]['raw_prompt']
            enhanced_prompt = f"{raw_prompt}. {reasoning_text}"
            enhanced_prompt_list.append(enhanced_prompt)
            
            messages = [{'role': 'user', 'content': [{'text': enhanced_prompt}]}]
            messages = self._transform_messages(messages)
            sft_format = self.processing_class.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            gen_prompt_list.append(sft_format)
        
        mesh_inputs = self.processing_class(
            text=gen_prompt_list,
            images=None, videos=None, 
            padding=True, return_tensors='pt'
        )
        
        mesh_ids, mesh_mask = mesh_inputs["input_ids"], mesh_inputs["attention_mask"]
        mesh_ids = mesh_ids.to(self.accelerator.device)
        mesh_mask = mesh_mask.to(self.accelerator.device)
        
        mesh_start_token_text = "<mesh-start>"
        try:
            mesh_start_tokens = self.processing_class.tokenizer.encode(mesh_start_token_text, add_special_tokens=False)
            print(f"Encoded mesh start tokens: {mesh_start_tokens}")
            
            if mesh_start_tokens and len(mesh_start_tokens) > 0:
                mesh_start_token_id = mesh_start_tokens[-1]  
                print(f"Using mesh start token from encoding: {mesh_start_token_id}")
            else:
                mesh_start_token_id = self.tokenizer.convert_tokens_to_ids("<|reserved_special_token_0|>")
                if mesh_start_token_id is None or mesh_start_token_id == self.tokenizer.unk_token_id:
                    mesh_start_token_id = self.tokenizer.eos_token_id
                print(f"Using fallback token: {mesh_start_token_id}")
        except Exception as e:
            print(f"Error encoding mesh start token: {e}")
            mesh_start_token_id = self.tokenizer.eos_token_id
            print(f"Using EOS token as fallback: {mesh_start_token_id}")
        
        try:
            mesh_ids = torch.cat([mesh_ids, mesh_ids.new_full((mesh_ids.size(0), 1), mesh_start_token_id)], dim=1)
            mesh_mask = torch.cat([mesh_mask, mesh_mask.new_ones((mesh_mask.size(0), 1))], dim=1)
            print(f"Successfully added mesh start token. New shape: {mesh_ids.shape}")
        except Exception as e:
            print(f"Error adding mesh start token: {e}")
            print(f"Continuing with original prompt shape: {mesh_ids.shape}")
        
        # new_generations_image
        mesh_ids = mesh_ids.repeat_interleave(self.new_generations_3d, dim=0)
        mesh_mask = mesh_mask.repeat_interleave(self.new_generations_3d, dim=0)
        mesh_inputs_embeds = unwrapped_model.get_input_embeddings()(mesh_ids)
        # check - grad - 
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            unwrapped_model.config.use_cache = False
            unwrapped_model.gradient_checkpointing_disable()
            
            eos_token_id = [self.processing_class.tokenizer.eos_token_id, 159858]
            total_generated_mesh_tokens = []
            split_size = 32
            
            for jj in range(0, mesh_ids.shape[0], split_size):
                print(f"Generating 3d model {jj}")
                start = jj
                end = min(jj + split_size, mesh_inputs_embeds.shape[0])
                cur_mesh_inputs_embeds = mesh_inputs_embeds[start: end]
                cur_mesh_mask = mesh_mask[start: end]
                
                mesh_completion_ids = unwrapped_model.generate(
                    inputs_embeds=cur_mesh_inputs_embeds,
                    attention_mask=cur_mesh_mask,
                    max_new_tokens=self.max_completion_length,
                    use_cache=True,
                    pad_token_id=self.processing_class.tokenizer.pad_token_id,
                    eos_token_id=self.processing_class.tokenizer.eos_token_id
                )
                
                original_prompt_len = cur_mesh_inputs_embeds.shape[1]
                total_generated_mesh_tokens.append(mesh_completion_ids)

        total_generated_mesh_tokens = torch.cat(total_generated_mesh_tokens, dim=0)
        print(f"Total new mesh tokens shape: {total_generated_mesh_tokens.shape}")
        
        model.module.gradient_checkpointing_enable()
        torch.set_grad_enabled(True)
        
  
        input_embeds = torch.cat(
            [
                model.module.get_input_embeddings()(mesh_ids),
                model.module.get_input_embeddings()(
                    total_generated_mesh_tokens
                )
            ],
            dim=1
        )
        attention_mask = torch.cat(
            [
                mesh_mask,
                torch.ones_like(total_generated_mesh_tokens)
            ],
            dim=1
        )
        
        per_token_logps = self._get_per_token_logps(
            model=model.module, 
            input_embeds=input_embeds,
            text_ids=None, 
            mesh_ids=total_generated_mesh_tokens, 
            attention_mask=attention_mask
        )

        prompt_length = mesh_ids.size(1)
        mesh_per_token_logps = per_token_logps[:, prompt_length - 1 :]
        mesh_completion_mask = torch.ones_like(total_generated_mesh_tokens)
        
        with torch.inference_mode():
            if self.ref_model is not None:
                self.ref_model.gradient_checkpointing_enable()
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, 
                    input_embeds=input_embeds,
                    text_ids=None, 
                    mesh_ids=total_generated_mesh_tokens, 
                    attention_mask=attention_mask
                )
                mesh_ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            else:
                # dummy ref_per_token_logps
                mesh_ref_per_token_logps = torch.zeros_like(per_token_logps)
                
        loss_dict['mesh-cot'] = {
            'per_token_logps': mesh_per_token_logps,
            'ref_per_token_logps': mesh_ref_per_token_logps,
            'completion_mask': mesh_completion_mask,
        }
        torch.set_grad_enabled(False)
        
        total_generated_mesh_tokens = total_generated_mesh_tokens.detach()
        mesh_tokens_list = []
        videos = []
        multi_view_images = []

        mesh_response = self.processing_class.tokenizer.batch_decode(total_generated_mesh_tokens, skip_special_tokens=False)
        
        current_step = getattr(self.state, 'global_step', 0) if hasattr(self, 'state') else 0
        
        for i in range(len(mesh_response)):
            mesh_tokens = self._parse_mesh_tokens(mesh_response[i])
            mesh_tokens_list.append(mesh_tokens)
            
            original_input_idx = i // (self.num_generations * self.new_generations_3d)
            generation_idx_in_group = i % (self.num_generations * self.new_generations_3d)
            raw_prompt = inputs[original_input_idx]['raw_prompt']
            prompt_id = raw_prompt[:50] if isinstance(raw_prompt, str) else f"prompt_{original_input_idx}"

            #video, video_org = self._generate_3d_model(mesh_tokens, enhanced_prompt_list[i])
            video, video_org = self._generate_3d_model(
                mesh_tokens, 
                enhanced_prompt_list[i],
                prompt_id=prompt_id,
                training_step=current_step,
                generation_idx=generation_idx_in_group
            )
            
            videos.append(video)

            views = []
            try:
                if video_org is not None and len(video_org) > 0:
                    num_frames = len(video_org)
                    num_views = self.vlm_reward_num_views
                    if num_frames < num_views:
                        indices = list(range(num_frames))
                    else:
                        indices = [int(j * (num_frames - 1) / (num_views - 1)) for j in range(num_views)]
                    for idx in indices:
                        views.append(video_org[idx])
            except Exception as _:
                views = []
            multi_view_images.append(views)


        prompts_for_reward = [input["raw_prompt"] for input in inputs for _ in range(self.num_generations) for _ in range(self.new_generations_3d)]
        
        reward_data = {
            'prompts': prompts_for_reward,
            'reasoning_texts': [self.processing_class.decode(completion_ids[i].cpu().tolist(), skip_special_tokens=True) 
                               for i in range(completion_ids.shape[0]) for _ in range(self.new_generations_3d)],
            'mesh_tokens_list': mesh_tokens_list,
            'multi_view_images': multi_view_images,
        }

        rewards_per_func = torch.zeros(len(prompts_for_reward), len(self.reward_funcs), device=self.accelerator.device)
        
        for i, (reward_func,reward_processing_class) in enumerate(zip(self.reward_funcs,self.reward_processing_classes)):
            if isinstance(reward_func, PreTrainedModel):

                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts_for_reward, reward_data['reasoning_texts'])]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts_for_reward, reward_data['reasoning_texts'])]
                
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                try:
                    reward_scores = reward_func.compute_reward(reward_data)
                    rewards_per_func[:, i] = torch.tensor(reward_scores, dtype=torch.float32, device=self.accelerator.device)
                except Exception as e:
                    print(f"Error computing reward with {reward_func.__name__}: {e}")

                    rewards_per_func[:, i] = torch.ones(len(prompts_for_reward), device=self.accelerator.device) * 0.5

        rewards = rewards_per_func.sum(dim=1)
        

        mean_grouped_rewards = rewards.view(-1, self.num_generations*self.new_generations_3d).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations*self.new_generations_3d).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations*self.new_generations_3d, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations*self.new_generations_3d, dim=0)

        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # if abs(advantages.sum()) < 1e-5:
        #     mean64 = advantages.to(torch.float64).mean()
        #     advantages = advantages - mean64.to(advantages.dtype) 
        
        torch.set_grad_enabled(True)
                
        for key in loss_dict['semantic-cot']:
            loss_dict['semantic-cot'][key] = loss_dict['semantic-cot'][key].repeat_interleave(self.new_generations_3d, dim=0)
        per_token_logps, ref_per_token_logps, completion_mask = [], [], []
        for key in ['semantic-cot', 'mesh-cot']:
            if loss_dict[key]['per_token_logps'] is None:
                loss_dict[key]['loss'] = None
                continue
            per_token_logps.append(loss_dict[key]['per_token_logps'])
            ref_per_token_logps.append(loss_dict[key]['ref_per_token_logps'])
            completion_mask.append(loss_dict[key]['completion_mask'])
        
        per_token_logps = torch.cat(per_token_logps, dim=1)
        ref_per_token_logps = torch.cat(ref_per_token_logps, dim=1)
        completion_mask = torch.cat(completion_mask, dim=1)
        
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()

        self._metrics[f"kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["loss"].append(self.accelerator.gather_for_metrics(loss.detach()).mean().item())
        
        completion_length = self.accelerator.gather_for_metrics(loss_dict['semantic-cot']['completion_mask'].sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()