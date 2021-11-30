from nilearn import signal
from brainspace.datasets import fetch_timeseries_preprocessing
from brainspace.datasets import load_confounds_preprocessing
import numpy as np
from nilearn import datasets
from brainspace.utils.parcellation import reduce_by_labels

timeseries = fetch_timeseries_preprocessing()
confounds_out = load_confounds_preprocessing()
clean_ts = signal.clean(timeseries.T, confounds=confounds_out).T

# Fetch surface atlas
atlas = datasets.fetch_atlas_surf_destrieux()

# Remove non-cortex regions
regions = atlas['labels'].copy()
masked_regions = [b'Medial_wall', b'Unknown']
masked_labels = [regions.index(r) for r in masked_regions]
for r in masked_regions:
    regions.remove(r)

# Build Destrieux parcellation and mask
labeling = np.concatenate([atlas['map_left'], atlas['map_right']])
mask = ~np.isin(labeling, masked_labels)

# Distinct labels for left and right hemispheres
lab_lh = atlas['map_left']
labeling[lab_lh.size:] += lab_lh.max() + 1

# extract mean timeseries for each label
seed_ts = reduce_by_labels(clean_ts[mask], labeling[mask], axis=1, red_op='mean')

#%%
from nilearn import plotting
from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')
correlation_matrix = correlation_measure.fit_transform([seed_ts.T])[0]
# Reduce matrix size, only for visualization purposes
mat_mask = np.where(np.std(correlation_matrix, axis=1) > 0.2)[0]
c = correlation_matrix[mat_mask][:, mat_mask]

# Create corresponding region names
regions_list = ['%s_%s' % (h, r.decode()) for h in ['L', 'R'] for r in regions]
masked_regions = [regions_list[i] for i in mat_mask]


corr_plot = plotting.plot_matrix(c, figure=(15, 15), labels=masked_regions,
                                 vmax=0.8, vmin=-0.8, reorder=True)


#%%
from brainspace.gradient import GradientMaps
from brainspace.datasets import load_fsa5, load_conte69
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

gm = GradientMaps(n_components=2, random_state=0)
gm.fit(correlation_matrix)

# Map gradients to original parcels
grad = [None] * 2
for i, g in enumerate(gm.gradients_.T):
    grad[i] = map_to_labels(g, labeling, mask=mask, fill=np.nan)


# Load fsaverage5 surfaces
surf_lh, surf_rh = load_fsa5()

plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1200, 400), cmap='viridis_r',
                 color_bar=True, label_text=['Grad1', 'Grad2'], zoom=1.5)




