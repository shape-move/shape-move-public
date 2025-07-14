<p align="center">
  <h1 align="center">Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions </h1>

  <div align="center">

🚩 Accepted by CVPR2025

<a href='https://arxiv.org/abs/2504.03639'><img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red'></a>
<a href='https://shape-move.github.io'><img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue'></a>


[Ting-Hsuan Liao](https://tinghliao.github.io/)<sup>1,2</sup>
[Yi Zhou](https://zhouyisjtu.github.io/)<sup>2</sup>
[Yu Shen](https://research.adobe.com/person/yu-shen/)<sup>2</sup>
[Chun-Hao Paul Huang](https://research.adobe.com/person/paulchhuang/)<sup>2</sup>
[Saayan Mitra](https://research.adobe.com/person/saayan-mitra/)<sup>2</sup>
[Jia-Bin Huang](https://jbhuang0604.github.io/)<sup>1</sup>
[Uttaran Bhattacharya](https://uttaranb127.github.io)<sup>2</sup>

<sup>1</sup>Adobe Research, <sup>2</sup>University of Maryland College Park
</div>

  <div align="center">
    <img src="data/images/teaser.png" alt="teaser" width="100%">
  </div>
  
</p> 

This is the implementation of ShapeMove, a framework for generating body-shape-aware human motion from text. ShapeMove combines a quantized VAE with continuous shape conditioning and a pretrained language model to synthesize realistic, shape-aligned motions from natural language descriptions.

## Enviroment setup

```
conda create -n shapemove python=3.10
conda activate shapemove
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## ⌛️ Download data

```
bash scripts/download_models.sh
```
This step will download pretrain model and SMPL model for visualization.


## 📐 Inference Model

```
bash scripts/demo.sh
```
The output motion and shape beta will be saved under `outputs`.

## 🎞️ Render Motion

### Blender setup
Follow the setup steps in [TEMOS](https://github.com/Mathux/TEMOS/tree/master?tab=readme-ov-file#rendering-motions-high_brightness).

After installing blender and required packages in the python environnement of blender.
Run the following command to ensure installation:
```
blender --background --version
```
Sould return Blender 2.93.18.

### Render meshes with Blender
```
# generate mesh with given beta and motion .npy file
python -m utils.mesh --dir [path/to/inference/output/folder]

# generate image from blender (with obj/ply file)
blender --background -noaudio --python utils/blender_render.py -- --mode=video --dir [path/to/mesh/folder]

# gather generated image and make video
python utils/visualization.py --dir [path/to/mesh/folder]
```


## Citation
```bibtex
@inproceedings{shapemove,
  title={Shape my moves: Text-driven shape-aware synthesis of human motions},
  author={Liao, Ting-Hsuan and Zhou, Yi and Shen, Yu and Huang, Chun-Hao Paul and Mitra, Saayan and Huang, Jia-Bin and Bhattacharya, Uttaran},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={1917--1928},
  year={2025}
}
```

## Acknowledgments 
Some great resources we benefit from: [MotionGPT](https://github.com/OpenMotionLab/MotionGPT/), [T2M-GPT](https://github.com/Mael-zys/T2M-GPT) and [text-to-motion](https://github.com/EricGuo5513/text-to-motion).