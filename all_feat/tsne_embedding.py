import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from pathlib import Path

# custom_colors = ['#86E3CE', '#D0E6A5', '#FFDD94', '#FA897B', '#CCABDB', '#DFA6FC']
custom_colors = ['#C5E7F7', '#D0E6A5', '#FFDD94', '#86E3CE', '#CCABDB', '#FA897B']

# 绘制 t-SNE 映射结果
def plot_tsne(features_tsne, labels, title, save_path):    
    # with plt.style.context("ggplot"):
    bwith = 1.5
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    for i in range(len(np.unique(labels))):
        plt.scatter(features_tsne[labels == i, 0], features_tsne[labels == i, 1], label=f'Class {i}',s=120, alpha=0.7)
        # c=custom_colors[i % len(custom_colors)], 
    #plt.title(title)
    plt.xticks([])
    plt.yticks([])
    #plt.legend()
    plt.savefig(save_path, dpi=400, bbox_inches='tight', pad_inches=0.1)
    plt.show()


for dataset in ['AVE']:
    for method in ['normal_fts','ogm_ge_fts','ume_fts','umt_fts','our_fts']:
        #### ours_cremad
        audio_features = np.load(f'/data/huacong/MMSA_Code/all_feat/{dataset}/{method}/audio_features.npy')
        visual_features = np.load(f'/data/huacong/MMSA_Code/all_feat/{dataset}/{method}/visual_features.npy')
        labels = np.load(f'/data/huacong/MMSA_Code/all_feat/{dataset}/{method}/all_labels.npy')

        print("Loaded audio", audio_features.shape)
        print("Loaded visual", visual_features.shape)
        print("Loaded labels", labels.shape)
        counts = np.bincount(labels)

        # 输出统计结果
        for i, count in enumerate(counts):
            print(f"Label {i}: {count} occurrences")

        # 使用 t-SNE 进行降维
        tsne = TSNE(n_components=2, random_state=42, n_iter=5000)

        audio_tsne = tsne.fit_transform(audio_features)
        visual_tsne = tsne.fit_transform(visual_features)
        
    
        savedir = Path('/data/huacong/MMSA_Code/all_save')
        savedir = savedir / dataset / method
        savedir.mkdir(parents=True, exist_ok=True)

        plot_tsne(audio_tsne, labels, "t-SNE Audio Features", os.path.join(savedir, 'audio.png'))
        plot_tsne(visual_tsne, labels, "t-SNE Visual Features", os.path.join(savedir, 'visual.png'))

