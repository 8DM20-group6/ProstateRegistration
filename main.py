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
ELASTIX_PATH = "C:\\elastix\\elastix.exe"
TRANSFORMIX_PATH = "C:\\elastix\\transformix.exe"
el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)

fixed_image_path = dataset.data_paths[0][0]
moving_image_path = dataset.data_paths[1][0]

el.register(
    fixed_image=fixed_image_path,
    moving_image=moving_image_path,
    parameters=[os.path.join("parameters", "parameters.txt")], # tune this, currently it is using b-spline transforms
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