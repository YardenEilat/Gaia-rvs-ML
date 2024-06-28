import random
import numpy as np

def generate_UMAP(leaves, num_neighbors=None, min_dist=None, n_components=2, metric='hamming', num_objects=50000):
    """generate UMAP embedding of the leaves"""
    import umap
    reducer = umap.UMAP(n_neighbors=num_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    if num_objects is not None:
        reducer.fit(random.sample(list(leaves), num_objects))
        embedding = reducer.transform(leaves)
    else:
        embedding = reducer.fit_transform(leaves)
    return embedding

def get_leaf_inds(rf_path='rvs_trained_rf_numtrees_500_highsnr.joblib', null_inds=[], num_objects=None):
    """loads RF model and returns leaf indices for each source"""
    import joblib
    d = np.load('filtered_nans_fluxes_and_ids_highsnr.npz', allow_pickle=True, mmap_mode='r+')
    fluxes = d['fluxes']
    fluxes = np.delete(fluxes, null_inds, axis=0)
    if num_objects is not None:
        fluxes = fluxes[:num_objects]
    rf_model = joblib.load(rf_path)
    leaves = rf_model.apply(fluxes)
    return leaves

def filter_low_snr(source_ids, threshold=50):
    """Remove low snr sources from the list
    output: indices with snr > threshold
    """
    with np.load('rvs_data_full_with_params3.npz', allow_pickle=True) as d:
        gaia_ids = d['gaia_id']
        rv_expected_sig_to_noise = d['rv_expected_sig_to_noise']
    rv_expected_sig_to_noise = rv_expected_sig_to_noise[[np.where(gaia_ids==source_id)[0][0] for source_id in source_ids]]
    return np.where(rv_expected_sig_to_noise>threshold)[0]


if __name__=="__main__":
    d = np.load('filtered_nans_fluxes_and_ids.npz', allow_pickle=True, mmap_mode='r+')
    fluxes = d['fluxes']
    source_ids = d['source_ids']
    snr_inds = filter_low_snr(source_ids)
    fluxes = fluxes[snr_inds]
    source_ids = source_ids[snr_inds]
    np.savez('filtered_nans_fluxes_and_ids_highsnr.npz', fluxes=fluxes, source_ids=source_ids)
    
    leaves = get_leaf_inds()
    X_embedded = generate_UMAP(leaves, num_neighbors=25, min_dist=0.5)
    np.savez('umap_embedding_full_25nn_highsnr.npz', X_embedded=X_embedded)
    
    