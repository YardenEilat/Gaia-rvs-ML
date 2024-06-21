import numpy as np
import pandas as pd

all_ids = pd.read_csv('blue_giant_group_inds.csv')
all_ids = all_ids.to_numpy()
separate_ids = pd.read_csv('blue_giants_separate_group_inds.csv')
separate_ids = separate_ids.to_numpy()
inter_ids = pd.read_csv('blue_giants_inter_group_inds.csv')
inter_ids = inter_ids.to_numpy()
# remove separate and inter from all to get main clust
main_clust_ids = np.setdiff1d(all_ids, separate_ids)
main_clust_ids = np.setdiff1d(main_clust_ids, inter_ids)

diff_of_separate = np.setdiff1d(separate_ids, all_ids)
new_separate = np.setdiff1d(separate_ids, diff_of_separate)
print(f'total amount is {len(all_ids)}')
print(f'separate amount is {len(separate_ids)}')
print(f'inter amount is {len(inter_ids)}')
print(f'main clust amount is {len(main_clust_ids)}')
print(f'diff of separate amount is {len(diff_of_separate)}')
print(f'new separate amount is {len(new_separate)}')

# pd.DataFrame(main_clust_ids).to_csv('blue_giants_main_clust_inds.csv', index=False)
# pd.DataFrame(diff_of_separate).to_csv('blue_giants_diff_of_separate_inds.csv', index=False)
# pd.DataFrame(new_separate).to_csv('blue_giants_new_separate_inds.csv', index=False)