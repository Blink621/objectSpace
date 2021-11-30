import pandas as pd
import numpy as np
from treelib import Tree
from os.path import join as pjoin

def gen_axes(parents, imagenet_hierarchy, neg_parents=None):
    """
    Get all children idx of defined parents name to generate axes

    Parameters
    ----------
    parent : list
        parent name in this axes

    Returns
    -------
    axes : ndarray
        1000 x 1

    """
    n_class = 1000
    axes = np.zeros((n_class))
    for parent_name in parents:
        parent_ID = imagenet_hierarchy.loc[imagenet_hierarchy.iloc[:, 0]==parent_name, 
                                           'classID'].values[0]
        child_class = tree.leaves(parent_ID)
        child_class = [x.identifier for x in child_class]
        axes[child_class] = 1
    # print(f'Find {int(axes.sum())} in this axes')
    if neg_parents != None:
        for parent_name in neg_parents:
            parent_ID = imagenet_hierarchy.loc[imagenet_hierarchy.iloc[:, 0]==parent_name, 
                                               'classID'].values[0]
            child_class = tree.leaves(parent_ID)
            child_class = [x.identifier for x in child_class]
            axes[child_class] = -1
    return axes

# define path
data_path = '/home/ubuntu/Project/data'
hierarchy_path = pjoin(data_path, 'hierarchy')
imagenet_hierarchy = pd.read_csv(pjoin(hierarchy_path, 'imagenet_hierarchy.csv'))

# generate tree
tree = Tree()
tree.create_node('entity', 1036)
for row in reversed(range(imagenet_hierarchy.shape[0])):
    tree.create_node(tag=imagenet_hierarchy.iloc[row, 0], 
                     identifier=imagenet_hierarchy.iloc[row, 2], 
                     parent=imagenet_hierarchy.iloc[row, 3],)
    print(f'Finish {row} row')

# get all kinds of trees
# define parent name
mobile_name = ['mammal', 'fungus', 'fish', 'bird', 'amphibian', 'reptile', 'invertebrate',
               'conveyance', 'person']
human_name = ['person']
animal_name = ['mammal', 'fungus', 'fish', 'bird', 'amphibian', 'reptile', 'invertebrate', 'person']
civilization_name = ['artifact', 'person']
nature_name = ['living_thing', 'fruit']
place_name = ['structure', 'geological_formation']
biological_name = ['living_thing', 'person', 'fruit']
# define axes
for hypothesized_name in ['mobile', 'human', 'animal', 'biological', 'place', 'civilization']:
    if hypothesized_name != 'civilization':
        tmp_axes = gen_axes(eval(f'{hypothesized_name}_name'), imagenet_hierarchy)
        print(f'Find {int(tmp_axes.sum())} in this {hypothesized_name} axes')
    else:
        tmp_axes = gen_axes(civilization_name, imagenet_hierarchy, nature_name)
    np.save(pjoin(data_path, 'hypothesized_axes', f'{hypothesized_name}_axes.npy'), tmp_axes)

#%% compute target hypothesized axes and pc coeff
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from os.path import join as pjoin
from scipy.stats import pearsonr

# load data
data_path = '/home/ubuntu/Project/data'
axes_path = pjoin(data_path, 'hypothesized_axes')
target = 'visual_area_with_ass'
hypothesized_names = ['mobile', 'human', 'animal', 'biological', 'place', 'civilization']
colors = ['black', 'red', 'green', 'blue']

n_pc = 4
brain_axes = np.load(pjoin(axes_path, f'brain_axes-{target}.npy'))[:n_pc, :]
hypothesized_corr = np.zeros((n_pc, len(hypothesized_names)))
for axes_idx, hypothesized_name in enumerate(hypothesized_names):
    tmp_axes = np.load(pjoin(axes_path, f'{hypothesized_name}_axes.npy'))
    for pc_idx in range(n_pc):
        hypothesized_corr[pc_idx, axes_idx] = pearsonr(tmp_axes, brain_axes[pc_idx])[0]**2

y_pos = np.arange(len(hypothesized_names))
plt.figure(figsize=(16, 8))
fig, axes = plt.subplots(1, 4)
font_title = {'family':'arial', 'weight':'bold', 'size':12}
font_label = {'family':'arial', 'weight':'bold', 'size':10}
for pc_idx in range(n_pc):
    data = hypothesized_corr[pc_idx, :]
    ax = axes[pc_idx]
    ax.set_xlim([0, 0.4])
    ax.set_xticklabels(np.linspace(0, 40, 3, dtype=int))
    # start plot
    ax.barh(y_pos, data, align='center', color=colors[pc_idx])
    if pc_idx == 0:
        ax.set_yticks(y_pos)
        ax.set_yticklabels(hypothesized_names, font_label)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_title(f'PC{pc_idx+1}', font_title)
fig.text(0.5, 0.005, '% Variance Explained', font_label, ha='center')
plt.tight_layout()
# save
result_path = '/home/ubuntu/Project/result'
plt.savefig(pjoin(result_path, f'compare_hypo-{target}.jpg'))
plt.close()


#%% Get best 5 and last 5 super class based on hierarchy dengram and cortex map correlation
# load data
from scipy.stats import pearsonr
import numpy as np
from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt

data_path = '/home/ubuntu/Project/data'
hierarchy_path = pjoin(data_path, 'hierarchy')
feature_path = pjoin(data_path, 'cortex_map')
axes_path = pjoin(data_path, 'hypothesized_axes')
surface_path = pjoin(data_path, 'visual_mask')
result_path = '/home/ubuntu/Project/result/pc_coef'
target = 'visual_area_with_ass'
special_flag = ''
decompose_method = 'pca'

class_mapping = pd.read_csv(pjoin(hierarchy_path, 'superClassMapping.csv'))
class_names = class_mapping['className']
classID = np.array(class_mapping['superClassID'])

class_labels = []
n_superclass = np.unique(classID).shape[0]
n_class_check = 10
n_pc = 4

# methods = ['pca', 'le', 'dm']
# for method in methods:
axes = np.load(pjoin(axes_path, f'brain_axes-{target}-{decompose_method}.npy')).T
axes_superclass = np.zeros((n_superclass, axes.shape[1]))
for idx,class_id in enumerate(np.unique(classID)):
    class_loc = classID == class_id
    axes_superclass[idx, :] = np.mean(axes[class_loc, :], axis=0)
    class_labels.append(class_mapping.loc[class_mapping['superClassID']==class_id, 
                                          'superClassName'].unique()[0])
class_labels[26] = 'gelo_form'

#%
for pc_id in range(n_pc):
    pc_single = axes_superclass[:, pc_id]
    # define big and small coef classes
    small_class = pc_single[np.argsort(pc_single)[:n_class_check]]
    small_class_name = np.array(class_labels)[np.argsort(pc_single)[:n_class_check]].tolist()
    big_class = pc_single[np.argsort(-pc_single)[:n_class_check]]
    big_class_name = np.array(class_labels)[np.argsort(-pc_single)[:n_class_check]].tolist()
    # plot figure
    y_pos = np.arange(n_class_check)
    plt.figure(figsize=(50, 22))
    fig, axes = plt.subplots(2, 1)
    font_title = {'family':'arial', 'weight':'bold', 'size':12}
    font_label = {'family':'arial', 'weight':'bold', 'size':7}
    # big class 
    # axes[0].set_xlim([0, 1])
    # axes[0].set_xticklabels(np.linspace(0, 80, 5, dtype=int))
    axes[0].bar(y_pos, big_class, color='red')
    axes[0].set_title(f'Top {n_class_check} high coef', font_title)
    axes[0].set_xticks(y_pos)
    axes[0].set_xticklabels(big_class_name, font_label)

    # small class 
    # axes[1].set_xlim([0, 0.4])
    # axes[1].set_xticklabels(np.linspace(0, 40, 3, dtype=int))
    axes[1].bar(y_pos, small_class, color='blue')
    axes[1].set_title(f'Top {n_class_check} low coef', font_title)
    axes[1].set_xticks(y_pos)
    axes[1].set_xticklabels(small_class_name, font_label)
    
    fig.text(0.5, 0.005, 'Superclass Name', font_label, ha='center')
    plt.tight_layout()
        
    plt.savefig(pjoin(result_path, target, decompose_method, 
                      f'class-{target}-{decompose_method}-PC{pc_id+1}.jpg'))
    plt.close()
    print(f'Finish PC{pc_id+1} in {target}')
        
        
        
        
