import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import itk
import elastix
import os
import random

class Registration():
    """
    Class to manage and visualize the registration process.
    """
    def __init__(self):
        """Initialize directories for storing results and parameters."""
        self.results_dir = "results"
        self.parameters_dir = "parameters"
    
    def select_data_paths(self, atlas_image_path, atlas_label_path, target_image_path, target_label_path=None, registration_name=0):
        """
        Select paths for input data.

        Args:
            atlas_image_path (str): Path to the moving image (atlas).
            atlas_label_path (str): Path to the segmentation of the moving image.
            target_image_path (str): Path to the fixed image (target).
            target_label_path (str, optional): Path to the segmentation of the fixed image.
            registration_name (int or str, optional): Name identifier for the registration process.

        """
        self.atlas_image_path = atlas_image_path
        self.atlas_label_path = atlas_label_path
        self.target_image_path = target_image_path
        self.target_label_path = target_label_path
        self.registration_name = registration_name
    
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
        self.result_transform_parameters.SetParameter(0, "FinalBSplineInterpolationOrder", "0")
        # TODO could look other interpolations, however this seems to work fine
        moving_image_transformix = itk.imread(atlas_label_path, itk.F)
        transformix_object = itk.TransformixFilter.New(moving_image_transformix)
        transformix_object.SetTransformParameterObject(self.result_transform_parameters)
        transformix_object.UpdateLargestPossibleRegion()
        
        # Save results
        self.atlas_label_deformed_image = transformix_object.GetOutput()
        self.atlas_label_deformed_path = os.path.join(
            self.results_dir, f"{self.registration_name}_atlas_label_deformed.mhd")
        itk.imwrite(self.atlas_label_deformed_image,
                    self.atlas_label_deformed_path)

    def perform_registration(self, parameter_file, plot):
        """Perform registration using specified parameter file.

        Args:
            parameter_file (str): Name of the parameter file for registration.
            plot (bool): Whether to plot registration process and results.
        """
        # Read images
        target_image = itk.imread(self.target_image_path, itk.F)
        atlas_image = itk.imread(self.atlas_image_path, itk.F)
        elastix_object = itk.ElastixRegistrationMethod.New(target_image, atlas_image)
        
        # Setup parameter object
        parameter_object = itk.ParameterObject.New()
        parameter_object.AddParameterFile(os.path.join(self.parameters_dir, parameter_file))
        elastix_object.SetParameterObject(parameter_object)
        
        # Perform registration
        elastix_object.SetOutputDirectory(self.results_dir)
        elastix_object.LogToFileOn()
        elastix_object.UpdateLargestPossibleRegion()
        
        # Save results
        self.result_transform_parameters = elastix_object.GetTransformParameterObject()
        self.atlas_image_deformed_image = elastix_object.GetOutput()
        self.atlas_image_deformed_path = os.path.join(
            self.results_dir, "result.0.mhd")
        # Only temporarily save deformed moving image 
        # (saving with code below gave errors with some parameter files)
        #     f"{self.registration_name}_atlas_image_deformed.mhd")
        # itk.imwrite(self.atlas_image_deformed_image,
        #             self.atlas_image_deformed_path)
        
        # Transform atlas label with registration result
        self.transform_atlas_label(self.atlas_label_path)

        if plot:
            self.plot_registration_process(parameter_file)
            self.plot_registration_results()

class MultiRegistrationFusion:
    """A class for performing multi-atlas registration and fusion of medical images."""
    def __init__(self, dataset, parameter_file, fusion_method="MajorityVoting", validation_results=None):
        """Initialize the MultiRegistrationFusion object.

        Args:
            dataset: The dataset containing image paths.
            parameter_file (str): Name of the parameter file for registration.
            fusion_method (str, optional): Fusion method for combining labels. Defaults to "MajorityVoting".
        """
        self.dataset = dataset
        self.parameter_file = parameter_file
        self.fusion_method = fusion_method
        if validation_results is None:
            self.validation_results = pd.DataFrame(columns=["parameter_file", "fusion_method", "target_index", "atlas_index", "dice"])
        else:
            self.validation_results = validation_results

    def perform_multi_atlas_registration(self, target_index, nr_atlas_registrations=4, validate=True):
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
        # Select N random atlas images to register onto target image
        atlas_indexes = random.sample(
            [idx for idx in range(len(self.dataset.data_paths)) if idx != target_index], nr_atlas_registrations)
        
        # Perform registration for each randomly selected atlas image
        for atlas_index in atlas_indexes:
            registration_name = f"{self.parameter_file[:-4]}_T{target_index}_A{atlas_index}"
            
            registration = Registration()
            registration.select_data_paths(atlas_image_path=self.dataset.data_paths[atlas_index][0],
                                            atlas_label_path=self.dataset.data_paths[atlas_index][1],
                                            target_image_path=self.dataset.data_paths[target_index][0],
                                            target_label_path=self.dataset.data_paths[target_index][1],
                                            registration_name=registration_name)
            
            registration.perform_registration(parameter_file=self.parameter_file, 
                                              plot=True)
            
            if validate:
                print("Validating performance of atlas label")
                dice = calculate_dsc(label1_path=registration.atlas_label_deformed_path,
                                    label2_path=registration.target_label_path, plot=True)
                results = pd.DataFrame([{"parameter_file": self.parameter_file, "fusion_method": self.fusion_method,
                          "target_index": target_index, "atlas_index": atlas_index, "dice": dice}])
                self.validation_results = pd.concat([self.validation_results, results], ignore_index=True)
        
        # Combine labels
        fused_atlas_label_path = self.label_fusion(
            atlas_indexes=atlas_indexes, target_index=target_index)
        
        if validate:
            print("Validating performance of fused atlas label")
            fused_dice = calculate_dsc(label1_path=fused_atlas_label_path,
                                label2_path=registration.target_label_path, plot=True)
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

    def label_fusion(self, atlas_indexes, target_index):
        """Fuse labels from registered atlas images.

        Args:
            atlas_indexes (list): Indexes of atlas images used for fusion.
            target_index (int): Index of the target image in the dataset.

        Returns:
            str: Path to the fused label.
        """
        fused_label = None
        individual_labels = []

        # TODO implement other fusion methods (look in literature or experiment)
        if self.fusion_method == "MajorityVoting":
            
            for atlas_index in atlas_indexes:
                # Load the deformed atlas label
                registration_name = f"{self.parameter_file[:-4]}_T{target_index}_A{atlas_index}_atlas_label_deformed.mhd"
                deformed_label_path = os.path.join("results", registration_name)
                deformed_label = itk.imread(deformed_label_path, itk.UC)

                # Store individual labels for plotting
                individual_labels.append(deformed_label)

                # Initialize fused label for first iteration
                if fused_label is None:
                    fused_label = np.zeros(deformed_label.shape, dtype=np.uint8)

                # Update fused label for majority voting, sum binary images
                fused_label += deformed_label

            # Apply majority voting rule, select part of label where 
            # at least half of the insividual labels have a value of 1
            fused_label[fused_label < (len(atlas_indexes) / 2)] = 0
            fused_label[fused_label >= (len(atlas_indexes) / 2)] = 1

        # Plot individual labels and fused label
        fig, axes = plt.subplots(
            1, len(individual_labels) + 1, figsize=(15, 5))
        for i, label in enumerate(individual_labels):
            axes[i].imshow(label[40, :, :], cmap='gray')
            axes[i].set_title(f"Atlas {i+1} Label")
            axes[i].axis('off')
        axes[-1].imshow(fused_label[40, :, :], cmap='gray')
        axes[-1].set_title("Fused Label")
        axes[-1].axis('off')
        plt.show()

        # Save the fused label
        fused_atlas_label_path = f"results/{self.parameter_file[:-4]}_T{target_index}_FusedLabel.mhd"
        itk.imwrite(itk.GetImageFromArray(fused_label), fused_atlas_label_path)
        return fused_atlas_label_path

# TODO could move these functions to a utils file
def img_from_path(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def overlay_from_segmentation(img_segmentation):
    return np.ma.masked_where(img_segmentation == 0, img_segmentation)
    
# Validation
def plot_dice(label1_img, label2_img, target_img_path=None):
    """Plot the overlap between two labels along with their individual regions.
    
    Args:
        label1_img (ndarray): First label/segmentation image.
        label2_img (ndarray): Second label/segmentation image.
    """
    # Set alpha channel to 0 for pixels with value 0
    label1_alpha = np.where(label1_img == 1, 1, 0)
    label2_alpha = np.where(label2_img == 1, 1, 0)
    overlap_alpha = np.where(np.logical_and(label1_img, label2_img) == 1, 1, 0)

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(label1_alpha[40, :, :], cmap='Blues')
    ax[0].set_title('Label 1')
    ax[0].axis('off')

    ax[1].imshow(label2_alpha[40, :, :], cmap='Reds')
    ax[1].set_title('Label 2')
    ax[1].axis('off')

    # TODO improve this plot
    ax[2].imshow(label1_alpha[40, :, :], cmap='Blues', alpha=0.8)
    ax[2].imshow(label2_alpha[40, :, :], cmap='Reds', alpha=0.8)
    ax[2].imshow(overlap_alpha[40, :, :], cmap='Purples')
    if target_img_path is not None:
        ax[2].imshow(img_from_path(target_img_path)[40, :, :], cmap='Purples')
    ax[2].set_title('Overlap between Label 1 and Label 2')
    ax[2].axis('off')

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

# TODO add hausdorff and other metrics