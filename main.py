"""
Atlas-based segmentation prostate

8DM20 - Group 6
O. Capmany Dalmas, P. Del Popolo, Z. Farshidrokh, D. Le, J. van der Pas, M. Vroemen
Utrecht University & University of Technology Eindhoven

"""
# %%

from dataset import Dataset
# %% Data loading and splitting
dataset = Dataset()
print(dataset.get_data_paths())
dataset.plot(0, slice=40)
img, mask = dataset[0]

# dataset.plot_data(1, slice=40)
# dataset.plot_data(3, slice=40)

# %% Registration
# %% Atlas transformation
# %% Atlas combining
# %% Validation
