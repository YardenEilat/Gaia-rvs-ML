from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import random

@njit
def distance_matrix(i, leaves, num_trees, nn):
    # dist = np.ones(NUM_NN) * 5.0
    dist = np.empty(len(leaves[:, 0]))
    num_trees = len(leaves[0,:])
    for j in range(len(leaves[:,1])):
        # num_trees = np.sum(leaf_labels[range(len(leaves[i,:])), leaves[i,:]]*leaf_labels[range(len(leaves[j,:])), leaves[j,:]])
        # num_leaves = np.sum((leaves[i,:]==leaves[j,:]) * leaf_labels[range(len(leaves[i,:])), leaves[i,:]])
        num_leaves = np.sum(leaves[i,:]==leaves[j,:])
        distance= 1-num_leaves/num_trees
        # distance = scipy.spatial.distance.hamming(leaves[i,:], leaves[j,:])
        # if distance < np.max(dist):
        #     dist[np.argmax(dist)] = distance
        dist[j] = distance
    return np.mean(np.sort(dist)[:nn])


def train_rf(train_features, train_labels, num_trees, *kwargs):
    """train random forest to classify between real and fake fluxes"""
    rf = RandomForestClassifier(n_estimators = num_trees, n_jobs=-1, random_state=42, verbose=1, min_samples_split=30, *kwargs)
    rf.fit(train_features, train_labels)
    joblib.dump(rf, f'rvs_trained_rf_numtrees_{num_trees}_highsnr.joblib')
    return rf


def get_leaf_labels(rf):
    """returns a list of the labels of each leaf node"""
    leaf_labels = []
    for tree in rf.estimators_:
        leaf_labels.append(tree.tree_.value[:, 0, 0])
    return np.array(leaf_labels)
    

def generate_UMAP(leaves, num_neighbors=None, min_dist=None, n_components=2, metric='hamming'):
    """generate UMAP embedding of the leaves"""
    import umap
    reducer = umap.UMAP(n_neighbors=num_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    embedding = reducer.fit_transform(leaves)
    return embedding     

def filter_low_snr(source_ids, threshold=50):
    """Remove low snr sources from the list
    output: indices with snr > threshold
    """
    with np.load('rvs_data_full_with_params3.npz', allow_pickle=True) as d:
        gaia_ids = d['gaia_id']
        rv_expected_sig_to_noise = d['rv_expected_sig_to_noise']
    rv_expected_sig_to_noise = rv_expected_sig_to_noise[[np.where(gaia_ids==source_id)[0][0] for source_id in source_ids]]
    return np.where(rv_expected_sig_to_noise>threshold)[0]
    
    
if __name__=='__main__':
    save_path = 'rf_parameters/'
    num_objs = 50000
    # num_trees_list = [250, 500, 1000]
    # max_depth_list = [4, 5, 6]
    min_samples_list = [None]
    num_trees_list = [500]
    nn = [10000]
    # max_leaf_nodes_list = [20]
    with np.load('rvs_fluxes_with_fakes_for_training_highsnr.npz') as f:
        total_fluxes = f['total_fluxes']
        is_true_flux = f['is_true_flux']
        source_ids = f['source_ids']
    # snr_indices = filter_low_snr(source_ids)
    # total_fluxes = np.concatenate((total_fluxes[is_true_flux][snr_indices], total_fluxes[~is_true_flux]))
    # is_true_flux = np.concatenate((is_true_flux[is_true_flux][snr_indices], is_true_flux[~is_true_flux]))
    # source_ids = source_ids[snr_indices]
    #run RF on data to teach to differentiate between real and fake flux

    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(total_fluxes, is_true_flux, test_size = 0.2, random_state = 42)
    random.seed(42)
    train_inds = random.sample(range(len(train_features[:,1])), min(num_objs, len(train_features[:,1])))
    train_features = train_features[train_inds,:]
    train_labels = train_labels[train_inds]
    test_inds = random.sample(range(len(test_features[:,1])), min(num_objs, len(test_features[:,1])))
    test_features = test_features[test_inds,:]
    test_labels = test_labels[test_inds]
    
    for num_trees in num_trees_list:
        rf = train_rf(train_features, train_labels, num_trees)
        predictions = rf.predict(test_features)
        accuracy = 1-sum(abs(predictions-test_labels))/len(test_labels)
        print('accuracy = ' + str(accuracy))
        print(f'mean prediction: {np.mean(predictions)}')
        pred = rf.predict(test_features[test_labels==1,:])
        print(f'percentage of real fluxes {np.mean(pred)}')
        import multiprocessing as mp
        from tqdm import tqdm
        print(f'cpu count is {mp.cpu_count()}')
        for num_neighbors in nn:
            weirdness_score = np.empty(len(test_features[test_labels==1,:]))
            weirdness_score[:] = np.nan
            batches=1
            leaves = rf.apply(test_features[test_labels==1,:])
            for i in range(batches):
                with mp.Pool(processes=mp.cpu_count()-1) as pool:
                    # dist_mat = pool.map(distance_matrix, tqdm(range(int(len(fluxes[:,1])))))
                    weirdness_score_temp = pool.map(partial(distance_matrix, leaves=leaves, num_trees=num_trees, nn=num_neighbors), tqdm(range(int(len(weirdness_score)/batches*i), int(len(weirdness_score)/batches*(i+1)))))
                weirdness_score[int(len(weirdness_score)/batches*i):int(len(weirdness_score)/batches*(i+1))] = np.array(weirdness_score_temp)
            print(f'weirdness mean={np.mean(weirdness_score)}, std={np.std(weirdness_score)}')
            plt.figure()
            plt.hist(weirdness_score, bins='auto', density=True)
            plt.title(f'trees={num_trees}, accuracy={accuracy:.2f}')
            plt.xlabel('weirdness score')
            plt.ylabel('object density')
            plt.savefig(f'{save_path}t={num_trees}_a={accuracy:.2f}_nn={num_neighbors}_highsnr.png')
            
        embedding = generate_UMAP(leaves, num_neighbors=int(len(leaves[:,0])/2000), min_dist=0.5, n_components=2, metric='hamming')
        plt.figure()
        plt.scatter(embedding[:,0], embedding[:,1], s=0.05)
        plt.title(f'UMAP t={num_trees}, accuracy={accuracy:.2f}')
        plt.savefig(f'{save_path}UMAP_t={num_trees}_a={accuracy:.2f}_highsnr.png')
            