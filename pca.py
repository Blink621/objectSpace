import os
import numpy as np
import pandas as pd
import igraph as ig
import random
import matplotlib.pyplot as plt
import scipy.io as sio
from os.path import join as pjoin
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA, DictionaryLearning, NMF
from scipy.stats import pearsonr

def plot_tree(coef, flag):
    """
    

    Parameters
    ----------
    coef : ndarray
        size: (1000,)
    flag : str
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # load data
    data_path = '/home/ubuntu/Project/data/hierarchy'
    result_path = '/home/ubuntu/Project/result/pc_coef'
    imagenet_hierarchy = pd.read_csv(pjoin(data_path, 'imagenet_hierarchy.csv'))
    # compute super class node coef
    super_class_names = imagenet_hierarchy['parent_name'].unique().tolist()
    n_node = imagenet_hierarchy['parentID'].max()+1
    n_class = 1000
    
    # get all nodes coefficient
    coef_all = np.zeros(n_node)
    coef_all[:n_class] = coef 
    cross_hierarchy_class = []
    for class_name in super_class_names:
        tmp_loc = imagenet_hierarchy.loc[imagenet_hierarchy['parent_name'] == \
                                         class_name, 'classID'].tolist()
        tmp_coef = coef_all[tmp_loc]
        # handle cases for cross hierarchy
        if 0 in tmp_coef:
            cross_hierarchy_class.append(class_name)
            continue
        tmp_ID = imagenet_hierarchy.loc[imagenet_hierarchy['parent_name'] == class_name, 
                                        'parentID'].tolist()[0]
        coef_all[tmp_ID] = np.mean(tmp_coef)
    
    for class_name in cross_hierarchy_class:
        tmp_loc = imagenet_hierarchy.loc[imagenet_hierarchy['parent_name'] == \
                                         class_name, 'classID'].tolist()
        tmp_coef = coef_all[tmp_loc]
        # handle cases for cross hierarchy
        if 0 in tmp_coef:
            print(f'Error in {class_name} in second search')
            continue
        tmp_ID = imagenet_hierarchy.loc[imagenet_hierarchy['parent_name'] == class_name, 
                                        'parentID'].tolist()[0]
        coef_all[tmp_ID] = np.mean(tmp_coef)
            
    # Generate graph now 
    g = ig.Graph()
    
    # Add vertices and edges first
    g.add_vertices(n_node)
    # define edges, to make eack super class is in a center of the class star
    edges = []
    for idx in range(imagenet_hierarchy.shape[0]):
        edges.append((imagenet_hierarchy.loc[idx, 'classID'], imagenet_hierarchy.loc[idx, 'parentID']))
    g.add_edges(edges)
    
    # Define vertex label names, only visualize super class names
    vertex_label = ['' if node_id < 1000 else imagenet_hierarchy.loc[imagenet_hierarchy['parentID'] == \
                    node_id, 'parent_name'].unique().tolist()[0]\
                    for node_id in range(n_node)]
    # These label will not visualize as they bind each other so close
    select_label_not_visualize = ['equipment', 'amphibian', 'plant','fungus',
                                  'toiletry', 'furnishing', 'entity'] 
    for label in select_label_not_visualize:
        vertex_label[vertex_label.index(label)] = ''
        
    # define visual style
    layout = g.layout('kk')
    times = abs(15/np.mean(np.abs(coef)))
    visual_style = {}
    visual_style["vertex_size"] = [abs(tmp_coef)*times for tmp_coef in coef_all]
    visual_style["vertex_color"] = ['red' if tmp_coef > 0 else 'blue' for tmp_coef in coef_all]
    # visual_style["vertex_shape"] = vertex_shape 
    visual_style["vertex_label"] = vertex_label
    visual_style["vertex_label_size"] = 15
    visual_style["vertex_label_color"] = 'white'
    visual_style["edge_width"] = np.hstack((np.repeat(0.5, n_class), 
                                            np.repeat(4, n_node-n_class+1)))
    visual_style["layout"] = layout
    visual_style["bbox"] = (1000, 1000)
    visual_style["margin"] = 50
    # save imageW
    random.seed(2613) # to make the same vertex location in different pc graph
    out_path = pjoin(result_path, flag.split('-')[-2], flag.split('-')[-1])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    ig.plot(g, pjoin(out_path, f'PC{flag}.png'), **visual_style, background=(0,0,0,1))
    random.seed(2613)

def plot_ratio(explained_variance_ratio, flag):
    
    """
    

    Parameters
    ----------
    coef : ndarray
        size: (1000,)
    flag : str
        DESCRIPTION.

    Returns
    -------
    None.
    """
    # plot variance explained graph
    x = np.arange(len(explained_variance_ratio))
    y = np.array(explained_variance_ratio)
    
    font_title = {'family':'serif', 'weight':'bold', 'size':18}
    font_label = {'family':'serif', 'weight':'bold', 'size':14}
    plt.figure(figsize=(8, 6))
    plt.title('Amount of Model Variance Explained', font_title)
    plt.xlabel('Principal Component', font_label)
    plt.ylabel('% Variance Explained', font_label)
    
    plt.plot(x, y, marker='o', ls='-', lw=0.5)
    plt.xticks(np.linspace(0, len(ratio)-1, len(ratio), dtype=int), 
               np.linspace(1, len(ratio), len(ratio), dtype=int),
               fontproperties='arial', weight='bold', size=10)
    high_point = 10
    plt.yticks(np.linspace(0.02, high_point/100, int((high_point-2)/2+1)), 
               np.linspace(2, high_point, int((high_point-2)/2+1), dtype=int), 
               fontproperties='arial', weight='bold', size=10)
    # plt.xscale('log', basex=2)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    # save
    result_path = '/home/ubuntu/Project/result/pc_coef'
    plt.savefig(pjoin(result_path, f'variance_ratio-{flag}.jpg'))
    plt.close()

def gen_transformer(decompose_method, n_components):
    """
     Get sklearn decomposition Class object

    Parameters
    ----------
    decompose_method : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    transformer = {'pca': PCA(n_components), 
                   'ica': FastICA(n_components, max_iter=8000, random_state=2613), 
                   'dl': DictionaryLearning(n_components, random_state=2613),
                   'nmf': NMF(n_components, init='random', random_state=2613),
                   }
    return transformer[decompose_method]
        
        
#%% main scripts

# load data
data_path = '/home/ubuntu/Project/data'
hierarchy_path = pjoin(data_path, 'hierarchy')
feature_path = pjoin(data_path, 'cortex_map')
axes_path = pjoin(data_path, 'hypothesized_axes')
target = 'whole_brain_select'
class_mapping = pd.read_csv(pjoin(hierarchy_path, 'superClassMapping.csv'))
class_names = class_mapping['className']
decompose_method = 'pca'
special_flag = 'transpose'

brain_axes_path = pjoin(axes_path, f'brain_axes-{target}-{special_flag}.npy')
if os.path.exists(brain_axes_path):
    brain_axes = np.load(brain_axes_path)
else:
    feature_matrix = np.load(pjoin(feature_path, f'sub-01-10_imagenet-feature-{target}.npy'))
    # feature_matrix = feature_matrix.transpose((1,0))
    # scale the data to make all feature positive in NMF
    if decompose_method == 'nmf':
        scaler = MinMaxScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
    # dimension reduction transforming
    n_components = 10
    transformer = gen_transformer(decompose_method, n_components)
    transformer.fit(feature_matrix)
    brain_axes = transformer.transform(feature_matrix)
    cortex_map = transformer.components_
    np.save(pjoin(feature_path, f'pc-cortex_map-{target}-{decompose_method}-{special_flag}.npy'), cortex_map)
    np.save(pjoin(axes_path, f'brain_axes-{target}-{decompose_method}-{special_flag}.npy'), brain_axes)
    # load hypothesized object axes space
    # animacy_org = sio.loadmat(pjoin(hierarchy_path, 'animate_or_not.mat'))['animate_label'].squeeze()
    # animacy = np.array([0 if x==-1 else x for x in animacy_org])
    # Compare hypothesized object axes space and data mining axes
    # pc1 = brain_axes[0, :]
    # corr = pearsonr(pc1, animacy)[0]
    # print(f'R2 in PC1: {corr**2}')

    # plot variance explained graph in 
    if decompose_method == 'pca':
        ratio = transformer.explained_variance_ratio_
        plot_ratio(ratio, target + '-' + special_flag)

#%% plot tree graph
n_pc = 4
if decompose_method == 'ica':
    n_pc = 8
for pc_idx in range(n_pc):
    pc_axes = brain_axes[:, pc_idx]
    plot_tree(pc_axes, f'{special_flag}-PC{pc_idx+1}-{target}-{decompose_method}')
    print(f'Finish Plotting PC{pc_idx+1} Graph')
    