import nibabel as nib
import numpy as np
from os.path import join as pjoin


def save_ciftifile(data, cortex_loc, filename):
    # prepare the 91k space
    voxel_all = np.zeros(91282)
    voxel_all[cortex_loc] = data
    # prepare save
    template = '/home/ubuntu/Project/data/visual_mask/template.dtseries.nii'
    ex_cii = nib.load(template)
    ex_cii.header.get_index_map(0).number_of_series_points = 1
    nib.save(nib.Cifti2Image(voxel_all.reshape((1,91282)),ex_cii.header), filename)

def sig_axes(axes, threshold):
    """
    Filter axes to make the selected voxels are significant in specialized threshold

    Parameters
    ----------
    axes : TYPE
        DESCRIPTION.
    threshold : TYPE
        DESCRIPTION.

    Returns
    -------
    axes_tmp : TYPE
        DESCRIPTION.

    """
    axes_tmp = np.zeros(axes.shape)
    outlier_loc = np.abs((axes - axes.mean())/axes.std()) > threshold
    axes_tmp[outlier_loc] = axes[outlier_loc]
    print(f'Find {outlier_loc.sum()} voxels significant in this axes')
    return axes_tmp

def discrete_axes(axes1, axes2):
    """
    Divided the two axes to four quadrant

    Parameters
    ----------
    axes1 : TYPE
        DESCRIPTION.
    axes2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    axes_discrete = np.zeros(axes1.shape)
    axes1 = axes1[:, np.newaxis]
    axes2 = axes2[:, np.newaxis]
    axes_sum = np.concatenate((axes1, axes2), axis=1)
    for idx in range(axes_sum.shape[0]):
        # localize the significant voxels and distinguish the quadrant
        if axes_sum[idx, 0] > 0 and axes_sum[idx, 1] > 0:
            axes_discrete[idx]  = 1
        elif axes_sum[idx, 0] < 0 and axes_sum[idx, 1] > 0:
            axes_discrete[idx]  = 2
        elif axes_sum[idx, 0] < 0 and axes_sum[idx, 1] < 0:
            axes_discrete[idx]  = 3
        elif axes_sum[idx, 0] > 0 and axes_sum[idx, 1] < 0:
            axes_discrete[idx]  = 4
    return axes_discrete


data_path = '/home/ubuntu/Project/data'
map_path = pjoin(data_path, 'cortex_map')
voxel_path = pjoin(data_path, 'voxel_coef')

target = 'visual_area_with_ass-pca'
cortex_coef = np.load(pjoin(map_path, f'pc-cortex_map-{target}.npy'))
cortex_loc = np.load(pjoin(map_path, 'sub-01-10_imagenet-visual_area_with_ass_idx.npy'))
n_pc = 4

# transfotm the voxel coef to become discrete
animacy = cortex_coef[:, 0]
PC2 = cortex_coef[:, 1]
PC3 = cortex_coef[:, 2]
size = cortex_coef[:, 3]

for threshold in [0, 1]:
    animacy = sig_axes(animacy, threshold)
    size = sig_axes(size, threshold)
    PC2 = sig_axes(PC2, threshold)
    PC3 = sig_axes(PC3, threshold)
    # 1st comparison: animacy vs size
    # animacy_vs_size = discrete_axes(animacy, size)
    # filename = pjoin(voxel_path, f'animacy_vs_size_sigma{threshold}.dtseries.nii')
    # save_ciftifile(animacy_vs_size, cortex_loc, filename)
    # 2nd comparison: size vs aspect_ratio
    size_vs_PC2 = discrete_axes(size, PC2)
    filename = pjoin(voxel_path, f'size_vs_PC2_sigma{threshold}.dtseries.nii')
    save_ciftifile(size_vs_PC2, cortex_loc, filename)

    size_vs_PC3 = discrete_axes(size, PC3)
    filename = pjoin(voxel_path, f'size_vs_PC3_sigma{threshold}.dtseries.nii')
    save_ciftifile(size_vs_PC3, cortex_loc, filename)
    
    # 3rd comparison: animacy vs aspect_ratio
    animacy_vs_PC2 = discrete_axes(animacy, PC2)
    filename = pjoin(voxel_path, f'animacy_vs_PC2_sigma{threshold}.dtseries.nii')
    save_ciftifile(animacy_vs_PC2, cortex_loc, filename)
    
    animacy_vs_PC3 = discrete_axes(animacy, PC3)
    filename = pjoin(voxel_path, f'animacy_vs_PC3_sigma{threshold}.dtseries.nii')
    save_ciftifile(animacy_vs_PC3, cortex_loc, filename)
