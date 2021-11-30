import nibabel as nib
import numpy as np
from os.path import join as pjoin


def save_ciftifile(data, filename):
    template = '/home/ubuntu/Project/data/visual_mask/template.dtseries.nii'
    ex_cii = nib.load(template)
    ex_cii.header.get_index_map(0).number_of_series_points = 1
    nib.save(nib.Cifti2Image(data.reshape((1,91282)),ex_cii.header), filename)



data_path = '/home/ubuntu/Project/data'
map_path = pjoin(data_path, 'cortex_map')
voxel_path = pjoin(data_path, 'voxel_coef')

target = 'visual_area_with_ass-ica'
cortex_coef = np.load(pjoin(map_path, f'pc-cortex_map-{target}.npy'))
cortex_loc = np.load(pjoin(map_path, 'sub-01-10_imagenet-visual_area_with_ass_idx.npy'))
n_pc = 8

for pc_idx in range(n_pc):
    voxel_all = np.zeros(91282)
    voxel_all[cortex_loc] = cortex_coef[:, pc_idx].squeeze()
    filename = pjoin(voxel_path, f'sub-01-10_imagenet-pc{pc_idx+1}-{target}.dtseries.nii')
    save_ciftifile(voxel_all, filename)
    print(f'Finish PC {pc_idx+1}')

