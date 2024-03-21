import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itk
import elastix
import os
import random
from LabelFusion.wrapper import fuse_images
import sklearn
from sklearn.metrics import normalized_mutual_info_score

class Registration():
    """
    Class to manage and visualize the registration process.
    """

    def __init__(self, atlas_image_path, atlas_label_path, target_image_path, parameter_file, target_label_path=None, registration_name=0):
        """
            atlas_image_path (str): Path to the moving image (atlas).
            atlas_label_path (str): Path to the segmentation of the moving image.
            target_image_path (str): Path to the fixed image (target).
            parameter_file (str): Name of the parameter file for registration.
            target_label_path (str, optional): Path to the segmentation of the fixed image.
            registration_name (int or str, optional): Name identifier for the registration process.

        """
        self.results_dir = "results"
        self.parameters_dir = "parameters"

        # Set input data paths
        self.atlas_image_path = atlas_image_path
        self.atlas_label_path = atlas_label_path
        self.target_image_path = target_image_path
        self.target_label_path = target_label_path
        
        # Registration parameters
        self.parameter_file = parameter_file
        self.registration_name = registration_name
        # define number of steps
        if type(self.parameter_file) == list:
            self.registration_setps = len(self.parameter_file)
        else:
            self.registration_setps = 1
    
    # Plotting
    def plot_registration_process(self):
        """Plot the registration process, metric vs iterations"""
        for i in range(self.registration_setps):
            title = self.parameter_file if self.registration_setps == 1 else self.parameter_file[i]
            plt.figure(figsize=(8, 4))
            # Iterate over files in the directory
            for filename in os.listdir(self.results_dir):
                if filename.startswith(f"IterationInfo.{i}.") and filename.endswith(".txt"):
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
        """Plot registration results."""
        # Get images and overlays
        images = [
            (self.atlas_image_path, self.atlas_label_path, "Atlas (moving) image"),
            (self.target_image_path, None, "Target (fixed) image"),
            (self.atlas_image_deformed_path,self.atlas_label_deformed_path, "Deformed atlas and label"),
            (self.target_image_path, self.atlas_label_deformed_path, "Target with deformed label")
        ]

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))
        for i, (image_path, label_path, title) in enumerate(images):
            row, col = divmod(i, 2)
            # Plot image
            img = img_from_path(image_path)
            ax[row, col].imshow(img[40, :, :], cmap="gray")
            # Plot overlay
            if label_path is not None:
                label_img = img_from_path(label_path)
                overlay_img = overlay_from_segmentation(label_img)
                ax[row, col].imshow(overlay_img[40, :, :],
                                    cmap="Wistia", alpha=0.6)
            ax[row, col].set_title(title)
            ax[row, col].axis('off')

        # Add target label overlay to bottom right subplot
        ax[1, 1].imshow(overlay_from_segmentation(img_from_path(
            self.target_label_path)[40, :, :]), cmap="spring", alpha=0.6)
        fig.suptitle("Atlas registration results")
        plt.tight_layout()
        plt.show()

    # Elastix functions    
    def transform_atlas_label(self, atlas_label_path):
        """Transform the atlas label based on the registration results.

        Args:
            atlas_label_path (str): Path to the label of the moving image (atlas).
        """
        # Setup transformation
        for i in range(self.registration_setps):
            # Need to set the interpolation order for every transform parameter to 0 for a binary image
            self.result_transform_parameters.SetParameter(i, "FinalBSplineInterpolationOrder", "0")
            
        moving_image_transformix = itk.imread(atlas_label_path, itk.F)
        transformix_object = itk.TransformixFilter.New(moving_image_transformix)
        transformix_object.SetTransformParameterObject(self.result_transform_parameters)
        transformix_object.UpdateLargestPossibleRegion()
        
        
        # Save results

        # Save results
        self.atlas_label_deformed_image = transformix_object.GetOutput()
        
        # Make sure output label is binary
        # self.atlas_label_deformed_image = sitk.BinaryThreshold(self.atlas_label_deformed_image, lowerThreshold=0.5, upperThreshold=1.0, insideValue=1, outsideValue=0)
        self.atlas_label_deformed_path = os.path.join(
            self.results_dir, f"{self.registration_name}_atlas_label_deformed.mhd")
        
        # Save results
        itk.imwrite(self.atlas_label_deformed_image,
                    self.atlas_label_deformed_path)

    def perform_registration(self, plot):
        """Perform registration using specified parameter file.

        Args:
            plot (bool): Whether to plot registration process and results.
        """
        # Read images
        target_image = itk.imread(self.target_image_path, itk.F)
        atlas_image = itk.imread(self.atlas_image_path, itk.F)
        elastix_object = itk.ElastixRegistrationMethod.New(target_image, atlas_image)
        
        # Setup parameter object
        parameter_object = itk.ParameterObject.New()
        if type(self.parameter_file) == list:
            # Registration is multistep if a list of parameter files is inputted
            for parameter in self.parameter_file:
                parameter_object.AddParameterFile(os.path.join(self.parameters_dir, parameter))
        else:
            parameter_object.AddParameterFile(os.path.join(self.parameters_dir, self.parameter_file))
        elastix_object.SetParameterObject(parameter_object)
        
        # Perform registration
        elastix_object.SetOutputDirectory(self.results_dir)
        elastix_object.LogToFileOn()
        elastix_object.UpdateLargestPossibleRegion()
        
        # Save results
        self.result_transform_parameters = elastix_object.GetTransformParameterObject()
        self.atlas_image_deformed_image = elastix_object.GetOutput()
        # self.atlas_image_deformed_path = os.path.join(
        #     self.results_dir, "result.0.mhd")

        
        self.atlas_image_deformed_path = os.path.join(
            self.results_dir, f"{self.registration_name}_atlas_image_deformed.mhd")
        itk.imwrite(self.atlas_image_deformed_image,
                    self.atlas_image_deformed_path)
        
        # Transform atlas label with registration result
        self.transform_atlas_label(self.atlas_label_path)

        if plot:
            self.plot_registration_process()
            self.plot_registration_results()
            

class MultiRegistrationFusion:
    """A class for performing multi-atlas registration and fusion of medical images."""
    def __init__(self, dataset, parameter_file, fusion_method="MajorityVoting", validation_results=None, plot=True):
        """Initialize the MultiRegistrationFusion object.

        Args:
            dataset: The dataset containing image paths.
            parameter_file (str): Name of the parameter file for registration.
            fusion_method (str, optional): Fusion method for combining labels. Defaults to "MajorityVoting".
        """
        self.dataset = dataset
        self.parameter_file = parameter_file
        self.fusion_method = fusion_method  # STAPLE, MayorityVoting, ITKVoting, SIMPLE
        if validation_results is None:
            self.validation_results = pd.DataFrame(columns=["parameter_file", "fusion_method", "target_index", "atlas_index", "dice"])
        else:
            self.validation_results = validation_results
        self.plot = plot
    
    def calculate_nmi(self, target_array, atlas_array):
        return normalized_mutual_info_score(target_array, atlas_array, average_method='arithmetic')

    def select_images_with_lowest_nmi(self, atlas_index, target_index, nr_atlas_registrations):
        #atlas_indexes = [idx for idx in range(atlas_index) if idx != target_index]

        # Calculate NMI scores for each pair of target and atlas images
        nmi_scores = []
        for atlas_idx in atlas_index:
            target_image = itk.imread(self.dataset.data_paths[target_index][0], itk.F)
            atlas_image = itk.imread(self.dataset.data_paths[atlas_idx][0], itk.F)
            # Convert images to arrays, this datatype is required to calculate the NMI scores
            target_array = itk.array_from_image(target_image).flatten()
            atlas_array = itk.array_from_image(atlas_image).flatten()
            nmi = self.calculate_nmi(target_array, atlas_array)
            #List of tuples containing the atlas index and the obtained NMI score for
            nmi_scores.append((atlas_idx, nmi))

        # Sort by NMI scores and select the first nr_atlas_registrations indices, corresponding to the highest NMI scores
        sorted_nmi_scores = sorted(nmi_scores, key=lambda x: x[1], reverse=True)
        selected_atlas_indices = [idx for idx, _ in sorted_nmi_scores[:nr_atlas_registrations]]
    
        return selected_atlas_indices

    def perform_multi_atlas_registration(self, target_index, nr_atlas_registrations=4, validate=True): #validate=True
        """Perform multi-atlas registration for a specific target image.
        1) Perform registration of N random moving (atlas) images onto one selected target image
        2) Save the deformed labels and optionally validate with the ground truth
        3) Fuse the deformed labels into one atlas label which is the predicted label for that target image

        Args:
            target_index (int): Index of the target image in the dataset.
            nr_atlas_registrations (int, optional): Number of atlas registrations to perform. Defaults to 4.
            validate (bool, optional): turn off validation mode if code is in testing or deployment mode or 
                no ground truth available.
        """

        # Select N atlas images to register onto target image based on NMI score
        atlas_full_list = [idx for idx in range(len(self.dataset.data_paths)) if idx != target_index]
        atlas_indexes = self.select_images_with_lowest_nmi(atlas_full_list, target_index, nr_atlas_registrations)

        #List that will contain all the deformed labels to fuse        
        labels_to_fuse = []

        # Perform registration for each randomly selected atlas image
        for atlas_index in atlas_indexes:

            registration_name = f"{self.parameter_file[:-4]}_T{target_index}_A{atlas_index}"
            
            registration = Registration(atlas_image_path=self.dataset.data_paths[atlas_index][0],
                                            atlas_label_path=self.dataset.data_paths[atlas_index][1],
                                            target_image_path=self.dataset.data_paths[target_index][0],
                                            target_label_path=self.dataset.data_paths[target_index][1],
                                            registration_name=registration_name,
                                            parameter_file=self.parameter_file)
            
            registration.perform_registration(plot=self.plot)
            
            if validate:
                print("Validating performance of atlas label")
                dice = calculate_dsc(label1_path=registration.atlas_label_deformed_path,
                                    label2_path=registration.target_label_path, plot=self.plot)
                results = pd.DataFrame([{"parameter_file": self.parameter_file, "fusion_method": self.fusion_method,
                          "target_index": target_index, "atlas_index": atlas_index, "dice": dice}])
                self.validation_results = pd.concat([self.validation_results, results], ignore_index=True)
            
            # Read the deformed labels one by one and store them in a list
            registered_label_image = sitk.ReadImage(registration.atlas_label_deformed_path, sitk.sitkUInt8)
            labels_to_fuse.append(registered_label_image)

        # Perform label fusion using the fused_simple strategy (modify as needed: STAPLE, MayorityVoting, ITKVoting, SIMPLE)
        if self.fusion_method == "SIMPLE":
            fused_result = fuse_images(
                labels_to_fuse, method=self.fusion_method, class_list=[0, 1])
        else:
            fused_result = fuse_images(
                labels_to_fuse, method=self.fusion_method)

        # Write the fused result to output file
        fused_atlas_label_path = f"results/{self.parameter_file[:-4]}_T{target_index}_FusedLabel.mhd"
        sitk.WriteImage(fused_result, fused_atlas_label_path)

        # Plot individual labels and fused label
        fig, axes = plt.subplots(1, len(labels_to_fuse) + 2, figsize=(15, 5))
        for i, label in enumerate(labels_to_fuse):
            axes[i].imshow(sitk.GetArrayFromImage(label[:, :, 40]), cmap='Reds', vmin=0, vmax=1)
            axes[i].contour(sitk.GetArrayFromImage(
                label[:, :, 40]), colors='black', linewidths=0.5)
            axes[-1].imshow(sitk.GetArrayFromImage(label[:, :, 40]), cmap='Reds', vmin=0, vmax=1.5, alpha=0.6)
            axes[-1].contour(sitk.GetArrayFromImage(label[:, :, 40]),
                             colors='black', linewidths=0.5)
            axes[i].set_title(f"Label {i+1}")
            axes[i].axis('off')
        axes[-2].imshow(sitk.GetArrayFromImage(fused_result[:, :, 40]), cmap='Blues', vmin=0, vmax=1)
        axes[-2].contour(sitk.GetArrayFromImage(fused_result[:,
                         :, 40]), colors='black', linewidths=0.5)
        axes[-2].set_title("Fused Label")
        axes[-2].axis('off')
        axes[-1].imshow(sitk.GetArrayFromImage(fused_result[:, :, 40]), cmap='Blues', vmin=0, vmax=1.5, alpha=0.8)
        axes[-1].contour(sitk.GetArrayFromImage(fused_result[:,
                         :, 40]), colors='black', linewidths=0.5)
        axes[-1].set_title("Combination plot")
        axes[-1].axis('off')
        plt.show()
        
        if validate:
            print("Validating performance of fused atlas label")
            fused_dice = calculate_dsc(label1_path=fused_atlas_label_path,
                                       label2_path=registration.target_label_path, plot=self.plot)
            results = pd.DataFrame([{"parameter_file": self.parameter_file, "fusion_method": self.fusion_method,
                      "target_index": target_index, "atlas_index": "fused_atlas", "dice": fused_dice}])
            self.validation_results = pd.concat([self.validation_results, results], ignore_index=True)
            return fused_atlas_label_path, self.validation_results
        
        else:
            return fused_atlas_label_path

    def full_multi_atlas_registration(self, nr_atlas_registrations=4, validate=True):
        """Perform multi-atlas registration for all target images in the dataset."""
        for target_index in range(len(self.dataset.data_paths)):
            _ = self.perform_multi_atlas_registration(nr_atlas_registrations, target_index, validate)
        return self.validation_results

# TODO could move these functions to a utils file
def img_from_path(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def overlay_from_segmentation(img_segmentation):
    return np.ma.masked_where(img_segmentation == 0, img_segmentation)
    
# Validation
def plot_dice(label1_img, label2_img, target_img_path=None, pixel_cutoff=50):
    """Plot the overlap between two labels along with their individual regions.
    
    Args:
        label1_img (ndarray): First label/segmentation image.
        label2_img (ndarray): Second label/segmentation image.
    """
    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(label1_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                  cmap='Reds', vmin=0, vmax=1)
    ax[0].contour(label1_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                   colors='black', linewidths=0.5)
    ax[0].set_title("Segmentation 1")
    ax[0].axis('off')

    ax[1].imshow(label2_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                  cmap='Blues', vmin=0, vmax=1)
    ax[1].contour(label2_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                   colors='black', linewidths=0.5)
    ax[1].set_title("Segmentation 2")
    ax[1].axis('off')

    ax[2].imshow(label1_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                  cmap='Reds', vmin=0, vmax=1.5, alpha=0.8)
    ax[2].imshow(label2_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                  cmap='Blues', vmin=0, vmax=1.5, alpha=0.6)
    ax[2].contour(label1_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                   colors='black', linewidths=0.5)
    ax[2].contour(label2_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                   colors='black', linewidths=0.5)
    ax[2].set_title('Both Segmentations')
    ax[2].axis('off')
    ax[2].set_facecolor('white')

    plt.tight_layout()
    plt.show()


def calculate_dsc(label1_path, label2_path, plot=False):
    label1_img = img_from_path(label1_path)
    label2_img = img_from_path(label2_path)
    
    overlap = np.multiply(label1_img, label2_img)
    dice = (2 * np.sum(overlap)) / (np.sum(label1_img) + np.sum(label2_img))
    print(f'The Dice score is {dice}')
    
    if plot:
        plot_dice(label1_img, label2_img)
        
    return dice

