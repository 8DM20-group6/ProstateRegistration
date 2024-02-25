"""
Atlas-based segmentation prostate

8DM20 - Group 6
O. Capmany Dalmas, P. Del Popolo, Z. Farshidrokh, D. Le, J. van der Pas, M. Vroemen
Utrecht University & University of Technology Eindhoven

"""
# %% Libraries
from dataset import Dataset
from registration import Registration
#%% Global variables
# Setup Elastix
# ELASTIX_PATH = "C:\\Users\\20203531\\Documents\\2. TUe\\8DM20\\Elastix\\elastix.exe"
# TRANSFORMIX_PATH = "C:\\Users\\20203531\\Documents\\2. TUe\\8DM20\\Elastix\\transformix.exe"
ELASTIX_PATH = "C:\\elastix\\elastix.exe"
TRANSFORMIX_PATH = "C:\\elastix\\transformix.exe"

# %% Registration functions

# TODO Atlas combining

# %%
parameter_files = [
    "parameters.txt",
    "Par0001affine.txt",
    "Par0001bspline04.txt",
    "Par0001bspline08.txt",
    "Par0001bspline16.txt",
    "Par0001bspline32.txt",
    "Par0001bspline64.txt",
    "Par0001rigid.txt",
    "Par0001translation.txt",
    "Par0043rigid.txt",
    "Par0055.txt"
]

# Data loading and splitting
dataset = Dataset()
target_index = 1
atlas_index = 2

# Registration
registration = Registration()
registration.select_data_paths(atlas_image_path=dataset.data_paths[atlas_index][0],
                               atlas_label_path=dataset.data_paths[atlas_index][1],
                               target_image_path=dataset.data_paths[target_index][0],
                               target_label_path=dataset.data_paths[target_index][1])

registration.perform_registration(parameter_file = parameter_files[0],
                                  plot = True)
#%%
dice = registration.calculate_dsc(label1_path=registration.atlas_label_deformed_image,
                                  label2_path=dataset.data_paths[target_index][1])

# %%
