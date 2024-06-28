import pandas as pd
import numpy as np
from tqdm import tqdm

astroparams = pd.read_pickle('astroparams_df.pkl')
d = np.load('filtered_nans_fluxes_and_ids_highsnr_no_duplicates.npz', allow_pickle=True, mmap_mode='r+')
source_ids = d['source_ids']

astroparam_ids = astroparams['source_id'].values
nan_row = pd.DataFrame(np.nan, index=[0], columns=astroparams.columns)
print(nan_row)
astroparams = astroparams.append(nan_row, ignore_index=True)

sorted_ids_inds = np.zeros_like(source_ids)
for i in tqdm(range(len(source_ids))):
    ind = np.where(source_ids[i] == astroparam_ids)[0]
    sorted_ids_inds[i] = ind[0] if np.any(ind) else -1

astroparams_sorted = astroparams.iloc[sorted_ids_inds]
astroparams_sorted.to_csv('astroparams_sorted_highsnr_no_duplicates.csv', index=False)
print(sorted_ids_inds)