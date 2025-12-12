# 🌟🔥 Are We Ready for RL in Text-to-3D Generation? A Progressive Investigation

Official repository for the paper "[Are We Ready for RL in Text-to-3D Generation? A Progressive Investigation](https://arxiv.org/pdf/2512.10949)".

[[📖 Paper](https://arxiv.org/pdf/2512.10949)] [[🤗 one-step Model](https://huggingface.co/IvanTang/3DGen-R1/tree/main)]

<p align="center">
    <img src="figures/teaser.png" width="100%"> <br>
</p>

## 💥 News
- **[2025.12.11]** We release the [checkpoint](https://huggingface.co/IvanTang/3DGen-R1/tree/main) of one-step AR3D-R1 ! 🔥
- **[2025.12.11]** We release the [arxiv paper](https://arxiv.org/pdf/2512.10949) and the evaluation code. 🔥

## 💪 Get Started
### Installation
Please set up the Python environment by:
```
conda env create -f environment.yml
conda activate environment_name

pip install -r requirements.txt
```

## 💫 Inference   
You can download the checkpoint from [here](https://huggingface.co/IvanTang/3DGen-R1/tree/main)
```
python demo.py
```

## 👀 Qualitative result

<p align="center">
    <img src="figures/visual1.png" width="90%"> <br>
</p>

<p align="center">
    <img src="figures/visual2.png" width="90%"> <br>
</p>
  
## 🗒️ TODO

- [ ] Release complete two-step training & evaluation code

- [ ] Release one-step training code

## 🧠 Related Work

- **[Image Generation CoT]** [Can We Generate Images with CoT? Let's Verify and Reinforce Image Generation Step by Step?](https://arxiv.org/pdf/2501.13926)
- **[T2I-R1]** [T2I-R1: Reinforcing Image Generation with Collaborative Semantic-level and Token-level CoT](https://arxiv.org/pdf/2505.00703)
- **[ShapeLLM-Omni]** [ShapeLLM-Omni: A Native Multimodal LLM for 3D Generation and Understanding](https://arxiv.org/abs/2506.01853)
- **[Trellis]** [Structured 3D Latents for Scalable and Versatile 3D Generation](https://arxiv.org/abs/2412.01506)
