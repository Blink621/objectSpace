import numpy as np
from os.path import join as pjoin
import pandas as pd
import matplotlib.pyplot as plt
from brainspace.datasets import load_conte69
from brainspace.plotting import plot_hemispheres, plot_surf
from brainspace.gradient import GradientMaps

# define path
data_path = '/home/ubuntu/Project/data'
hierarchy_path = pjoin(data_path, 'hierarchy')
map_path = pjoin(data_path, 'cortex_map')
surface_path = pjoin(data_path, 'visual_mask')
result_path = '/home/ubuntu/Project/result/gradient_map'
target = 'visual_area_with_ass'
method = 'pca'
special_flag = '-transpose'
n_pc = 4

# load data
cortex_map = np.load(pjoin(map_path, f'pc-cortex_map-{target}-{method}{special_flag}.npy'))
vertex_point = np.load(pjoin(surface_path, 'visual_area_32k_map.npy'))
if cortex_map.shape[0] != 19742:
    cortex_map = cortex_map.T
    
# load surface map
surf_lh, surf_rh = load_conte69()

# Merge data and transform surface space
cortex_32k_map = np.zeros((vertex_point.shape[0], n_pc))
for pc_idx in range(n_pc):
    pc_map = cortex_map[:, pc_idx]
    vertex_point[np.where(np.isnan(vertex_point)==False)[0]] = pc_map 
    cortex_32k_map[:, pc_idx] = vertex_point
cortex_32k_map = [cortex_32k_map[:, x] for x in range(n_pc)]

# plot_hemispheres(surf_lh, surf_rh, array_name=cortex_32k_map, size=(1200, 800), 
#                  color_bar=True, label_text=[f'Grad{x+1}' for x in range(n_pc)], zoom=1.5, 
#                  nan_color=(0,0,0,0.5), cmap='seismic', background=(1,1,1), transparent_bg=False, 
#                  screenshot=True, filename=pjoin(result_path, f'gradient_map-{target}-{method}.jpg'))
    
surfs = {'lh': surf_lh, 'rh': surf_rh}
layout = ['lh', 'rh', 'rh', 'lh']
view = ['lateral', 'medial', 'ventral', 'ventral']
share = 'r'

array_name=cortex_32k_map
layout = [layout] * len(array_name)
array_name2 = []
n_pts_lh = surf_lh.n_points
for an in array_name:
    if isinstance(an, np.ndarray):
        name = surf_lh.append_array(an[:n_pts_lh], at='p')
        surf_rh.append_array(an[n_pts_lh:], name=name, at='p')
        array_name2.append(name)
    else:
        array_name2.append(an)
array_name = np.asarray(array_name2)[:, None]

plot_surf(surfs, layout=layout, array_name=array_name, size=(1000, int(n_pc*200)), zoom=1.2, view=view, 
          color_bar=True, label_text=[f'Grad{x+1}' for x in range(n_pc)], share=share, #color_range =(-2.72, 2.72), 
          nan_color=(0,0,0,0.5), cmap='seismic', background=(1,1,1), transparent_bg=False, 
          screenshot=True, filename=pjoin(result_path, f'gradient_map-{target}-{method}{special_flag}.jpg'))
