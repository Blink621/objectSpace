import numpy as np
from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt

# load data
data_path = '/home/ubuntu/Project/data'
hierarchy_path = pjoin(data_path, 'hierarchy')
feature_path = pjoin(data_path, 'cortex_map')
axes_path = pjoin(data_path, 'hypothesized_axes')
target = 'visual_area_with_ass'
class_mapping = pd.read_csv(pjoin(hierarchy_path, 'superClassMapping.csv'))
feature_matrix = np.load(pjoin(feature_path, f'sub-01-10_imagenet-feature-{target}.npy'))
feature_matrix = feature_matrix.transpose((1,0))

# compute RDM
voxel_rdm = np.corrcoef(feature_matrix)
class_rdm = np.corrcoef(feature_matrix, rowvar=False)

#%% SHOW RAW RDM DATA
flag = 'voxel'
rdm = eval(f'{flag}_rdm')
np.fill_diagonal(rdm, 0)

# Plot figure of these correlations
f, ax = plt.subplots(1,1, figsize=(8, 7))

plt.imshow(
    rdm,
    cmap='jet', 
    vmin=rdm.min(),
    vmax=0.8,
)
plt.colorbar()
ax.set_title(f'{flag} RDM') 
ax.set_xlabel(f'{flag} id')
ax.set_ylabel(f'{flag} id')
result_path = '/home/ubuntu/Project/result/RDM'
plt.savefig(pjoin(result_path, f'raw_rdm-{flag}.jpg'))
plt.close()

#%% PCA on voxel RDM
from sklearn.decomposition import PCA

n_components = 4
cortex_map = PCA(n_components).fit_transform(voxel_rdm)
brain_axes = PCA(n_components).fit_transform(class_rdm)
np.save(pjoin(feature_path, 'pc-cortex_map-RDM.npy'), cortex_map)
np.save(pjoin(axes_path, 'brain_axes-RDM.npy'), brain_axes)

#%% Show Sorted Class RDM
super_class_id = np.array(class_mapping['superClassID'])
feature_matrix_sorted = np.zeros(feature_matrix.shape)
class_labels = []

class_flag = 0
for class_id in range(super_class_id.max()):
    class_loc = super_class_id == class_id+1
    class_num = np.sum(class_loc)
    feature_matrix_sorted[:, class_flag:class_flag+class_num] = feature_matrix[:, class_loc]
    class_flag += class_num
    class_labels.append(class_mapping.loc[class_mapping['superClassID']==class_id+1, 
                                          'superClassName'].unique()[0])

class_rdm_sorted = np.corrcoef(feature_matrix_sorted, rowvar=False)
np.fill_diagonal(class_rdm_sorted, 0)

#%% Plot Sorted
class_not_show = ['plant', 'amphibian', 'person', 'toiletry', 'abstraction', 'fungus',
                  'substance', 'feline', 'bar']
class_labels = ['' if x in class_not_show else x for x in class_labels]

f, ax = plt.subplots(1,1, figsize=(10, 8))
plt.imshow(
    class_rdm_sorted,
    cmap='jet', 
    vmin=-0.2,
    vmax=0.2,
)
plt.colorbar()
font_label = {'family':'arial', 'weight':'bold', 'size':9}

# Pull out the bin edges between the different categories
binsize = np.histogram(np.sort(super_class_id), 30)[0]
edges = np.concatenate([np.asarray([0]), np.cumsum(binsize)])[:-1]
ax.set_xticks(list(np.array(edges)))
ax.set_xticklabels(class_labels, font_label, rotation = 30)
ax.set_yticks(list(np.array(edges)))
ax.set_yticklabels(class_labels, font_label)
# ax.vlines(edges,0,1000)
# ax.hlines(edges,0,1000)
# ax.set_title('RSM, sorted, %s' % roi_names[roi_id])
result_path = '/home/ubuntu/Project/result/RDM'
plt.savefig(pjoin(result_path, 'sorted_rdm.jpg'))
plt.close()

#%%






