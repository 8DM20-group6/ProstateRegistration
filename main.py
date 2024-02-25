"""
Atlas-based segmentation prostate

8DM20 - Group 6
O. Capmany Dalmas, P. Del Popolo, Z. Farshidrokh, D. Le, J. van der Pas, M. Vroemen
Utrecht University & University of Technology Eindhoven

"""
# %% Libraries
import elastix
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import itk
from dataset import Dataset

#%% Global variables
# Setup Elastix
# ELASTIX_PATH = "C:\\Users\\20203531\\Documents\\2. TUe\\8DM20\\Elastix\\elastix.exe"
# TRANSFORMIX_PATH = "C:\\Users\\20203531\\Documents\\2. TUe\\8DM20\\Elastix\\transformix.exe"
ELASTIX_PATH = "C:\\elastix\\elastix.exe"
TRANSFORMIX_PATH = "C:\\elastix\\transformix.exe"

# %% Registration functions
class Registration():
    def __init__(self):
        self.results_dir = "results"
        self.parameters_dir = "parameters"
    
    def select_data_paths(self, atlas_image_path, atlas_label_path, target_image_path, target_label_path):
        self.atlas_image_path = atlas_image_path  # moving image
        self.atlas_label_path = atlas_label_path  # moving segmentation
        self.target_image_path = target_image_path  # fixed image
        self.target_label_path = target_label_path
    
    def img_from_path(self, img_path):
        return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

    def overlay_from_segmentation(self, img_segmentation):
        return np.ma.masked_where(img_segmentation == 0, img_segmentation)
    
    # Plotting
    def plot_registration_process(self, title):
        """Plot the registration process, metric vs iterations

        Args:
            title (str): parameter files used for registration
        """
        plt.figure(figsize=(8, 4))
        # Iterate over files in the directory
        for filename in os.listdir(self.results_dir):
            if filename.startswith("IterationInfo.0.") and filename.endswith(".txt"):
                filepath = os.path.join(self.results_dir, filename)
                log = elastix.logfile(filepath)

                itnrs = log['itnr']
                metrics = log['metric']

                plt.plot(itnrs, metrics, label=filename)

        # Set axis labels and title
        plt.xlabel('Iterations')
        plt.ylabel('Metric Value')
        plt.title(title)

        plt.legend()
        plt.show()
    
    def plot_registration_results(self):
        atlas_label_image = self.img_from_path(self.atlas_label_path)
        atlas_label_overlay = self.overlay_from_segmentation(atlas_label_image)
        atlas_label_deformed_image = self.overlay_from_segmentation(self.transform_atlas_label(self.atlas_label_path))
        atlas_label_deformed_overlay = self.overlay_from_segmentation(atlas_label_deformed_image)

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        images = [
            (self.atlas_image_path, atlas_label_overlay, "Atlas (moving) image"),
            (self.target_image_path, None, "Target (fixed) image"),
            (self.atlas_image_deformed_path, atlas_label_deformed_overlay, "Deformed atlas and label"),
            (self.target_image_path, atlas_label_deformed_overlay, "Target with deformed label")
        ]

        for i, (image_path, overlay_img, title) in enumerate(images):
            row = i // 2
            col = i % 2
            ax[row, col].imshow(self.img_from_path(image_path)[40, :, :], cmap="gray")
            if overlay_img is not None:
                ax[row, col].imshow(overlay_img[40, :, :], cmap="Wistia", alpha=0.6)
            ax[row, col].set_title(title)
            ax[row, col].axis('off')
        ax[1,1].imshow(self.overlay_from_segmentation(self.img_from_path(self.target_label_path)[40,:,:]), cmap="spring", alpha=1)
        fig.suptitle("Atlas registration results")
        plt.tight_layout()
        plt.show()
    
    # Elastix functions    
    def transform_atlas_label(self, atlas_label_path):
        self.result_transform_parameters.SetParameter(0, "FinalBSplineInterpolationOrder", "0")
        moving_image_transformix = itk.imread(atlas_label_path, itk.F)
        transformix_object = itk.TransformixFilter.New(moving_image_transformix)
        transformix_object.SetTransformParameterObject(self.result_transform_parameters)
        transformix_object.UpdateLargestPossibleRegion()
        self.atlas_label_deformed_image = transformix_object.GetOutput()
        return self.atlas_label_deformed_image

    def perform_registration(self, parameter_file, plot):
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(os.path.join(self.parameters_dir, parameter_file))
        fixed_image = itk.imread(self.target_image_path, itk.F)
        moving_image = itk.imread(self.atlas_image_path, itk.F)
        elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
        elastix_object.SetParameterObject(parameter_object)
        elastix_object.SetOutputDirectory(self.results_dir)
        elastix_object.LogToFileOn()
        elastix_object.UpdateLargestPossibleRegion()
        self.result_image = elastix_object.GetOutput()
        self.result_transform_parameters = elastix_object.GetTransformParameterObject()
        self.atlas_image_deformed_path = os.path.join(self.results_dir, "result.0.mha")
        itk.imwrite(self.result_image, self.atlas_image_deformed_path)
        
        if plot:
            self.plot_registration_process(parameter_file)
            self.plot_registration_results()
    
    def plot_dice(self, label1_path, label2_path):
        """Plot the overlap between two labels along with their individual regions.
        
        Args:
            label1_path (str): Path to the first label/segmentation image.
            label2_path (str): Path to the second label/segmentation image.
        """
        # Load label images
        # label1_image = self.img_from_path(label1_path)
        label1_image = label1_path
        label2_image = self.img_from_path(label2_path)
        print(np.unique(label1_image), np.unique(label2_image))
        
        # Compute overlap
        overlap = np.multiply(label1_image, label2_image)
        
        # Plotting
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))        
        ax[0].imshow(label1_image[40, :, :], cmap='gray')
        ax[0].set_title('Label 1')
        ax[0].axis('off')

        ax[1].imshow(label2_image[40, :, :], cmap='gray')
        ax[1].set_title('Label 2')
        ax[1].axis('off')

        cmap = cm.get_cmap('viridis', 3)
        ax[2].imshow((label1_image + 2 * label2_image +
                3 * overlap)[40, :, :], cmap=cmap)
        ax[2].set_title('Overlap between Label 1 and Label 2')
        ax[2].axis('off')
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([1, 2, 3])
        cbar = plt.colorbar(mappable, ax=ax[2])
        cbar.set_ticks([1, 2, 3])
        cbar.set_ticklabels(['Label 1', 'Label 2', 'Overlap'])
    
        plt.tight_layout()
        plt.show()
    
    # Validation
    def calculate_dsc(self, label1_path, label2_path):
        # label1_image = self.img_from_path(label1_path)
        label1_image = label1_path
        label2_image = self.img_from_path(label2_path)
        overlap = np.multiply(label1_image, label2_image)
        dice = (2*np.sum(overlap))/(np.sum(label1_image)+np.sum(label2_image))
        print(f'The Dice score is {dice}')
        self.plot_dice(label1_path, label2_path)
        return dice

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
dice = registration.calculate_dsc(label1_path = registration.atlas_label_deformed_image,
                                  label2_path=dataset.data_paths[target_index][1])

# %%
