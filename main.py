"""
Atlas-based segmentation prostate

8DM20 - Group 6
O. Capmany Dalmas, P. Del Popolo, Z. Farshidrokh, D. Le, J. van der Pas, M. Vroemen
Utrecht University & University of Technology Eindhoven

"""
# %%
import elastix
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import numpy as np
from dataset import Dataset
# %% Data loading and splitting
dataset = Dataset()
print(dataset.get_data_paths())
dataset.plot(0, slice=40)
img, mask = dataset[0]

# dataset.plot_data(1, slice=40)
# dataset.plot_data(3, slice=40)

# %% Registration
# Setup Elastix
ELASTIX_PATH = "C:\\Users\\20203531\\Documents\\2. TUe\\8DM20\\Elastix\\elastix.exe"
TRANSFORMIX_PATH = "C:\\Users\\20203531\\Documents\\2. TUe\\8DM20\\Elastix\\transformix.exe"
# ELASTIX_PATH = "C:\\elastix\\elastix.exe"
# TRANSFORMIX_PATH = "C:\\elastix\\transformix.exe"
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

# Select parameter file
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
parameter_file = parameter_files[1]

# Select fixed and moving images
fixed_image_path = dataset.data_paths[0][0]
moving_image_path = dataset.data_paths[1][0]

# Define registration
el.register(
    fixed_image=fixed_image_path,
    moving_image=moving_image_path,
    parameters=[os.path.join("parameters", parameter_file)],
    output_dir="results")

# %% Atlas transformation
tr = elastix.TransformixInterface(parameters=os.path.join("results", "TransformParameters.0.txt"),
                                  transformix_path=TRANSFORMIX_PATH)
warped_atlas_path = tr.transform_image(dataset.data_paths[0][1], output_dir="results")
warped_itk_atlas = sitk.ReadImage(os.path.join("results", "result.mhd"))
warped_atlas_array = sitk.GetArrayFromImage(warped_itk_atlas)

warped_itk_image = sitk.ReadImage(os.path.join("results", "result.0.mhd"))
warped_image_array = sitk.GetArrayFromImage(warped_itk_image)
fig, ax = plt.subplots(1, 2)
ax[0].imshow(warped_image_array[40,:,:], cmap="gray")
ax[1].imshow(warped_atlas_array[40,:,:], cmap="gray")
fig.suptitle("Warped atlas")
# %% Atlas combining
# %% Validation

def calculate_dsc(array1, array2):
    overlap = np.multiply(array1, array2)
    return (2*np.sum(overlap))/(np.sum(array1)+np.sum(array2))

print(f'The Dice score is {calculate_dsc(warped_atlas_array, dataset[1][1])}')

# %%