# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List
from datasets import load_dataset, load_from_disk
import torch

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from open_r1.trainer import JanusT2IR1Trainer3D
import ipdb

@dataclass
class GRPOConfig3D(GRPOConfig):
    new_generations_3d: int = field(default=1, metadata={"help": "The number of new generations of 3D model to generate"})
    mesh_token_num_per_model: int = field(default=1024, metadata={"help": "The number of mesh tokens to generate per 3D model"})
    vqvae_model_path: str = field(default=None, metadata={"help": "The path to the VQVAE model for 3D generation"})
    trellis_model_path: str = field(default="JeffreyXiang/TRELLIS-text-xlarge", metadata={"help": "The path to the Trellis model"})
    simplify: float = field(default=0.95, metadata={"help": "Mesh simplification factor"})
    texture_size: int = field(default=1024, metadata={"help": "Texture resolution for 3D models"})
    reasoning_prompt_path: Optional[str] = field(
        default='',
        metadata={"help": "Path to the reasoning prompt template"}
    )
    max_textcot_length: int = field(default=None, metadata={"help": "The maximum length of the text cot"})
    video_save_interval: int = field(default=10, metadata={"help": "Save video every N training steps (default: 10)"})
    
    # VLM-based multi-view reward controls
    vlm_reward_joint_multiview: bool = field(default=False, metadata={"help": "Feed multiple views jointly to VLM for a single strict decision."})
    vlm_reward_strict_all_views: bool = field(default=False, metadata={"help": "When not using joint multiview, require all views pass (AND) instead of majority vote."})
    vlm_reward_num_views: int = field(default=6, metadata={"help": "Number of views sampled from render/video for reward evaluation."})
    
    # vLLM API
    vlm_api_base: str = field(default="http://localhost:8000/v1", metadata={"help": "vLLM API base URL"})
    vlm_api_model_name: str = field(default="Qwen/Qwen2.5-VL-7B-Instruct", metadata={"help": "Model name for vLLM API (as configured in vLLM server)"})
    vlm_reward_temperature: float = field(default=0.1, metadata={"help": "Temperature for VLM reward generation"})
    
    # UnifiedReward API
    unified_api_base: str = field(default="http://localhost:8090/v1", metadata={"help": "UnifiedReward vLLM API base URL"})
    unified_api_model_name: str = field(default="UnifiedReward", metadata={"help": "Model name for UnifiedReward API"})
    unified_reward_num_views: int = field(default=6, metadata={"help": "Number of views for UnifiedReward evaluation"})
    unified_reward_temperature: float = field(default=0.0, metadata={"help": "Temperature for UnifiedReward generation"})
    unified_aggregate_method: str = field(default="mean", metadata={"help": "Aggregation method for multi-view scores: 'mean', 'min', or 'max'"})

    hps_ckpt_path: str = field(default=None, metadata={"help": "The path to the hps checkpoint"})
    epsilon_low: float = field(default=0.2, metadata={"help": "Lower bound for asymmetric clipping (DAPO style)"})
    epsilon_high: float = field(default=0.28, metadata={"help": "Upper bound for asymmetric clipping (DAPO style)"})
    
    # CLIP Text Reward
    clip_model_path: str = field(default="openai/clip-vit-large-patch14", metadata={"help": "CLIP model path for text-image alignment"})
    clip_reward_num_views: int = field(default=6, metadata={"help": "Number of views for CLIP reward evaluation"})
    clip_aggregate_method: str = field(default="max", metadata={"help": "Aggregation method for CLIP multi-view scores: 'mean', 'min', or 'max'"})
    clip_score_scale: float = field(default=5.0, metadata={"help": "Score scaling factor for CLIP reward"})
    
    # Two-stage CoT configuration
    enable_two_stage_cot: bool = field(default=False, metadata={"help": "Enable two-stage CoT training"})
  
@dataclass
class GRPOScriptArguments3D(ScriptArguments):
    reward_funcs: List[str] = field(
        default_factory=lambda: ["mesh_quality", "geometry", "texture"],
        metadata={"help": "List of 3D reward functions. Possible values: 'mesh_quality', 'geometry', 'texture'"},
    )

def make_3d_conversation(example):
    reasoning_prompt = f"""Let me think step by step about generating the 3D model for {example["prompt"]}.
    
I need to consider:
1. Overall geometric structure: What is the basic shape and form of this object?
2. Key features and details: What distinctive elements define this object?  
3. Proportions and scale: How do different parts relate in size and position?
4. Material properties: What surface textures and materials are needed?
5. Functional aspects: How does form follow function for this object?
6. Spatial relationships: How do components connect and interact in 3D space?
7. Topology considerations: What is the surface topology and mesh structure?
8. Rendering properties: How should this object appear when rendered?

My detailed reasoning process:
"""

    return {
        "prompt": [
            {"role": "user", "content": reasoning_prompt},
            {"role": "assistant", "content": ""},
        ],
        'raw_prompt': example["prompt"],
        'task_type': example.get('task_type', '3d_generation'),
    }

def make_3d_conversation_with_reference(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": f"Generate a 3D model based on this description: {example['prompt']}",
                "images": [example.get('reference_image', None)] if example.get('reference_image') else None
            },
            {"role": "assistant", "content": ""},
        ],
        'raw_prompt': example['prompt'],
        'reference_image': example.get('reference_image', None),
    }


reward_funcs_registry_3d = {
    "vlm_3d": 'vlm_3d',
    "hps": 'hps',
    "unified": 'unified', 
    "clip_text": 'clip_text',
}

def main(script_args, training_args, model_args):
    """3D GRPO"""
    
    reward_funcs = [reward_funcs_registry_3d[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.csv'):
        suffix = 'csv'
    elif script_args.dataset_name.endswith('.json'):
        suffix = 'json'
    elif script_args.dataset_name.endswith('.parquet'):
        suffix = 'parquet'
    dataset = load_dataset(suffix, data_files=script_args.dataset_name)
    print('Dataset length: ', len(dataset['train']))

    if training_args.reasoning_prompt_path:
        with open(training_args.reasoning_prompt_path, 'r') as f:
            cot_prompt = f.read()
            training_args.cot_prompt = cot_prompt
    
    if "reference_image" in dataset[script_args.dataset_train_split].features:
        print("***************has reference images in dataset***************")
        dataset = dataset.map(make_3d_conversation_with_reference)
    else:
        dataset = dataset.map(
            make_3d_conversation,
            num_proc=1,
        )
    
    trainer_cls = JanusT2IR1Trainer3D
    print("using: ", trainer_cls)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        script_args=script_args,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments3D, GRPOConfig3D, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
