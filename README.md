# ReconBoost
This is the official code for the paper  “ReconBoost: Boosting Can Achieve Modality Reconcilement.” accepted by International Conference on Machine Learning (ICML2024). This paper is available at [here](https://arxiv.org/abs/2405.09321).

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2405.09321) [![video](https://img.shields.io/badge/Video-Presentation-F9D371)](https://github.com/huacong/ReconBoost) [![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](https://github.com/huacong/ReconBoost) [![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://github.com/huacong/ReconBoost)

**Paper Title: ReconBoost: Boosting Can Achieve Modality Reconcilement.**   

**Authors: Cong Hua,  [Qianqian Xu$^{*}$](https://qianqianxu010.github.io/), [Shilong Bao](https://statusrank.github.io/), [Zhiyong Yang](https://joshuaas.github.io/), [Qingming Huang$^{*}$](https://people.ucas.ac.cn/~qmhuang)**   



<img src="docs\pipeline.png" alt="pipeline" style="zoom:67%;" />

## Installation

Clone this repository

```bash
git clone https://github.com/huacong/ReconBoost.git
```

Install the required libraries

```
pip install -r requirements.txt
```

## Dataset

In our paper, six benchmarks are adopted for evaluation. CREMA-D, AVE and ModelNet40 (front-rear views) are two-modality datasets and MOSI, MOSEI and SIMS are tri-modality datasets. . The statistics of all datasets used in the experiments are included in the table below.

![image-20240520001124087](docs\dataset.png)

## Training

For training, we provide hyper-parameter settings, running command and checkpoints for each dataset. To enhance the stability of the training process, we load the [**pre-trained uni-modal model**](cache\pretrained_unimodel) via specifying hyper-parameter `--use_pretrain`.

Get started with TensorBoard to monitor the training process.

```bash
tensorboard --logdir ./ --port 6007 --bind_all
```

The well-trained models are saved at [**here**](cache\ckpt). 

**CREMA-D dataset**

```bash
python train2.py 
--dataset CREMAD 
--dataset_path /data/huacong/CREMA/data
--n_class 6
--batch_size 64
--boost_rate 1.0
--n_worker 8
--epochs_per_stage 4
--correct_epoch 4
--use_lr True
--m_lr 0.01
--e_lr 0.01
--weight1 5.0
--weight2 1.0
--use_pretrain 
--m1ckpt modality1_ckpt_pth
--m2ckpt modality2_ckpt_pth
```

**AVE dataset**

```bash
python train2.py 
--dataset AVE 
--dataset_path /data/huacong/AVE_Dataset
--n_class 28
--batch_size 64
--boost_rate 1.0
--n_worker 8
--epochs_per_stage 4
--correct_epoch 4
--use_lr True
--m_lr 0.01
--e_lr 0.01
--weight1 4.0
--weight2 1.0
--use_pretrain 
--m1ckpt modality1_ckpt_pth
--m2ckpt modality2_ckpt_pth
```

**ModelNet40 dataset**

```bash
python train2.py
--dataset MView40
--dataset_path /data/huacong/ModelNet40
--n_class 40
--batch_size 48
--boost_rate 1.0
--n_worker 8
--epochs_per_stage 4
--correct_epoch 4
--weight1 4.0
--weight2 1.0
--use_pretrain 
--m1ckpt modality1_ckpt_pth
```

**MOSEI**

```bash
python train_MSA.py 
--dataset MSA
--dataset_name mosei
--featurePath /data/huacong/MSA/MOSEI/Processed/unaligned_50.pkl
--seq_lens [50, 1, 1]
--feature_dims [768, 74, 35]
```

**MOSI**

```bash
python train_MSA.py 
--dataset MSA
--dataset_name mosi
--featurePath /data/huacong/MSA/MOSI/Processed/unaligned_50.pkl
--seq_lens [50, 1, 1]
--feature_dims [768, 5, 20]
```

**SIMS**

```bash
python train_MSA.py 
--dataset MSA
--dataset_name sims
--featurePath /data/huacong/MSA/SIMS/unaligned_39.pkl
--seq_lens [50, 1, 1]
--feature_dims [768, 33, 709]
```

## 4. Evaluation 

#### 4.1 Overall Evaluation

**CREMA-D dataset**

```bash
python eval.py --dataset CREMAD --dataset_path /data/huacong/CREMA/data --n_class 6 --batch_size 64 --n_worker 8 
--ensemble_ckpt_path cache/ckpt/CREMAD/best_ensemble_net_XX.path 
--uni_ckpt_path cache/ckpt/CREMAD/uni_encoder_XX.pth
```

**AVE** 

```bash
python eval.py --dataset AVE --dataset_path /data/huacong/AVE_Dataset --n_class 28 --batch_size 64 --n_worker 8 --ensemble_ckpt_path cache/ckpt/AVE/best_ensemble_net_XX.path 
--uni_ckpt_path cache/ckpt/AVE/uni_encoder_XX.pth
```

#### 4.2 Uni-modal Linear-prob Evaluation

Evaluate the audio encoder on CREMA-D dataset.

```bash
python uni_eval.py --dataset CREMAD --dataset_path /data/huacong/CREMA/data --modality audio --n_class 6 --batch_size 64 --max_epochs 100 --emb 512 --uni_ckpt_path cache/ckpt/CREMAD/uni_encoder_XX.pth
```

Evaluate the visual encoder on AVE dataset.

```bash
python uni_eval.py --dataset AVE --dataset_path /data/huacong/AVE_Dataset --modality visual --n_class 28 --batch_size 64 --max_epochs 100 --emb 512 --uni_ckpt_path cache/ckpt/AVE/uni_encoder_XX.pth
```



#### 4.3 Latent Embedding Visualization

Latent embeddings among different competitors are saved in  .

To visualize high-dimension embedding, you can run the following command.

```bash
python tsne_embedding.py
```



## Citation

If you find this repository useful in your research, please cite the following papers:

```
@misc{hua2024reconboost,
  title={ReconBoost: Boosting Can Achieve Modality Reconcilement}, 
  author={Cong Hua and Qianqian Xu and Shilong Bao and Zhiyong Yang and Qingming Huang},
  year={2024},
  eprint={2405.09321},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

## Contact us

If you have any detailed questions or suggestions, you can email us: huacong23z@ict.ac.cn. We will reply in 1-2 business days. Thanks for your interest in our work!
