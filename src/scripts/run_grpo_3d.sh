cd gen3d-r1/src
RUN_NAME=""

export DEBUG_MODE="true"
export LOG_PATH="./outputs/debug_3d.txt"
export WANDB_API_KEY=""
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=$NO_PROXY 

MODEL_PATH="./models/ShapeLLM-7B-omni"  
HF_DATASET="./data/sampled_prompt.json" 
OUTPUT_DIR="outputs/${RUN_NAME}"
HPS_CKPT="./src/gen3d-r1/reward_weight/HPS_v2.1_compressed.pt"

VQVAE_PATH="./models/3DVQVAE/3DVQVAE.bin"
TRELLIS_PATH="./models/TRELLIS-text-xlarge"
REASONING_PROMPT_PATH="./data/prompt/3d_reasoning_prompt.txt" 

CUDA_VISIBLE_DEVICES=0 vllm serve ./src/gen3d-r1/reward_weight/UnifiedReward-2.0-qwen-7b \
--host 127.0.0.1 \
--trust-remote-code \
--served-model-name UnifiedReward \
--gpu-memory-utilization 0.85 \
--tensor-parallel-size 1 \
--pipeline-parallel-size 1 \
--limit-mm-per-prompt image=16 \
--max-num-seqs 8 \
--port 8090 > unifiedreward.log 2>&1 &

sleep 60


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=^lo,docker0
export NCCL_IB_DISABLE=1 

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES="1,2,3,4,5,6,7" \
torchrun --nproc_per_node="7" \
--nnodes="1" \
--node_rank="0" \
--master_addr="127.0.0.1" \
--master_port="12349" \
open_r1/grpo_3d.py --use_vllm False \
--deepspeed "../configs/zero3.json" \
--output_dir $OUTPUT_DIR \
--model_name_or_path $MODEL_PATH \
--dataset_name $HF_DATASET \
--max_prompt_length 1024 \
--max_completion_length 2048 \
--temperature 1.0 \
--num_generations 8 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 2 \
--logging_steps 1 \
--bf16  \
--torch_dtype bfloat16 \
--report_to wandb \
--gradient_checkpointing false \
--attn_implementation flash_attention_2 \
--max_steps 900 \
--run_name $RUN_NAME \
--save_steps 200 \
--new_generations_3d 1 \
--mesh_token_num_per_model 1024 \
--reasoning_prompt_path $REASONING_PROMPT_PATH \
--reward_funcs hps unified \
--beta 0.01 \
--tf32 true \
--learning_rate 1e-6 \
--vlm_reward_joint_multiview \
--vqvae_model_path $VQVAE_PATH \
--trellis_model_path $TRELLIS_PATH \
--hps_ckpt_path $HPS_CKPT \
--simplify 0.95 \
--texture_size 1024 \
--video_save_interval 200 \
--unified_api_base "http://127.0.0.1:8090/v1" \
--unified_api_model_name "UnifiedReward" \
--unified_reward_num_views 6 \
--unified_reward_temperature 0.1 \
--unified_aggregate_method "max" \
--enable_two_stage_cot False \