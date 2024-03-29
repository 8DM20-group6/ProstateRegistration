import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itk
import elastix
import os
from LabelFusion.wrapper import fuse_images
from skimage.metrics import normalized_mutual_information
import seg_metrics.seg_metrics as sg

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
        plt.tight_layout(w_pad=-20)
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

    def __init__(self, data_atlas_paths, data_target_path, parameter_file, fusion_method="MajorityVoting", validation_results=None, plot=True):
        """Initialize the MultiRegistrationFusion object.

        Parameters:
            data_atlas_paths: The data_atlas_paths containing image paths.
            parameter_file (str): Name of the parameter file for registration.
            fusion_method (str, optional): Fusion method for combining labels. Defaults to "MajorityVoting".
        """
        self.data_atlas_paths = data_atlas_paths
        self.data_target_path = data_target_path
        self.parameter_file = parameter_file
        self.fusion_method = fusion_method  # STAPLE, MayorityVoting, ITKVoting, SIMPLE
        if validation_results is None:
            self.validation_results = pd.DataFrame()
        else:
            self.validation_results = validation_results
        self.plot = plot

    def atlas_selection(self, atlas_indexes, nr_atlas_registrations):
        """
        Selects a specified number of atlas indices based on the normalized mutual information (NMI) scores
        between the target image and each atlas image.

        Parameters:
            atlas_indexes (list): A list of atlas indices.
            nr_atlas_registrations (int): The number of atlas registrations to select.

        Returns:
            list: A list of selected atlas indices.

        """

        # Calculate NMI scores for each pair of target and atlas images
        nmi_scores = []
        for atlas_index in atlas_indexes:
            target_image = itk.imread(self.data_target_path[0], itk.F)
            atlas_image = itk.imread(self.data_atlas_paths[atlas_index][0], itk.F)
            # Convert images to arrays, this datatype is required to calculate the NMI scores
            target_array = itk.array_from_image(target_image)
            atlas_array = itk.array_from_image(atlas_image)
            nmi = normalized_mutual_information(target_array, atlas_array)
            #List of tuples containing the atlas index and the obtained NMI score for
            nmi_scores.append((atlas_index, nmi))
        
        # Sort by NMI scores and select the first nr_atlas_registrations indices, corresponding to the highest NMI scores
        sorted_nmi_scores = sorted(nmi_scores, key=lambda x: x[1], reverse=True)
        selected_atlas_indices = [idx for idx, _ in sorted_nmi_scores[:nr_atlas_registrations]]
    
        return selected_atlas_indices

    def perform_multi_atlas_registration(self, nr_atlas_registrations=4, validate=True): #validate=True
        """Perform multi-atlas registration for a specific target image.
        1) Perform registration of N random moving (atlas) images onto one selected target image
        2) Save the deformed labels and optionally validate with the ground truth
        3) Fuse the deformed labels into one atlas label which is the predicted label for that target image

        Args:
            nr_atlas_registrations (int, optional): Number of atlas registrations to perform. Defaults to 4.
            validate (bool, optional): turn off validation mode if code is in testing or deployment mode or 
                no ground truth available.
        """

        # Select N atlas images to register onto target image based on NMI score
        atlas_indexes_full = list(range(len(self.data_atlas_paths)))
        atlas_indexes = self.atlas_selection(atlas_indexes_full, nr_atlas_registrations)

        #List that will contain all the deformed labels to fuse        
        labels_to_fuse = []
        
        target_name = self.data_target_path[0].split("\\")[-2]

        # Perform registration for each randomly selected atlas image
        for atlas_index in atlas_indexes:
            atlas_name = self.data_atlas_paths[atlas_index][0].split("\\")[-2]
            registration_name = f"Target-{target_name}_Atlas-{atlas_name}"
            
            registration = Registration(atlas_image_path=self.data_atlas_paths[atlas_index][0],
                                            atlas_label_path=self.data_atlas_paths[atlas_index][1],
                                            target_image_path=self.data_target_path[0],
                                            target_label_path=self.data_target_path[1],
                                            registration_name=registration_name,
                                            parameter_file=self.parameter_file)
            
            registration.perform_registration(plot=self.plot)
            
            if validate:
                print(f"Computing metrics for atlas label {atlas_name} with target label {target_name}")
                dice, hd, hd95, recall, fpr, fnr = compute_metrics(label1_path=registration.atlas_label_deformed_path,
                                    label2_path=registration.target_label_path, plot=self.plot)
                
                results = pd.DataFrame([{"parameter_file": self.parameter_file, "fusion_method": self.fusion_method, 
                                         "target": target_name, "atlas": atlas_name,
                                         "dice": dice, "hd": hd, "hd95": hd95, "recall": recall, 
                                         "fpr": fpr, "fnr": fnr}])
                
                self.validation_results = pd.concat([self.validation_results, results], ignore_index=True)
            
            # Read the deformed labels one by one and store them in a list
            registered_label_image = sitk.ReadImage(registration.atlas_label_deformed_path, sitk.sitkUInt8)
            labels_to_fuse.append(registered_label_image)
        
        # Label fusion using the fused_simple strategy (methods: STAPLE, MayorityVoting, ITKVoting, SIMPLE)
        if self.fusion_method == "SIMPLE":
            fused_result = fuse_images(labels_to_fuse, method=self.fusion_method, class_list=[0, 1])
        else:
            fused_result = fuse_images(labels_to_fuse, method=self.fusion_method)

        # Write the fused result to output file
        fused_atlas_label_path = f"results/Target-{target_name}_FusedLabel.mhd"
        sitk.WriteImage(fused_result, fused_atlas_label_path)

        # Plot individual labels and fused label
        if (self.plot==True) or (self.plot=="final"):
            self.plot_fusion(labels_to_fuse, fused_result)
        
        if validate:
            print(f"Computing metrics for the fused atlas label with target label {target_name}")
            fused_dice, hd, hd95, recall, fpr, fnr = compute_metrics(label1_path=fused_atlas_label_path,
                                       label2_path=registration.target_label_path, plot=self.plot)
            
            results = pd.DataFrame([{"parameter_file": self.parameter_file, "fusion_method": self.fusion_method, "NrOfAtlases": nr_atlas_registrations,
                                     "target": target_name, "atlas": "fused_atlas", "dice": fused_dice,
                      "hd":hd, "hd95": hd95, "recall": recall, "fpr": fpr, "fnr": fnr}])
            
            self.validation_results = pd.concat([self.validation_results, results], ignore_index=True)

        return fused_atlas_label_path
    
    def plot_fusion(self, labels_to_fuse, fused_result):
        """Plots of seperate and fused altas labels"""
        fig, axes = plt.subplots(1, len(labels_to_fuse) + 2, figsize=(15, 5))
        for i, label in enumerate(labels_to_fuse):
            axes[i].imshow(sitk.GetArrayFromImage(label[:, :, 40]), cmap='Reds', vmin=0, vmax=1)
            axes[i].contour(sitk.GetArrayFromImage(label[:, :, 40]), colors='black', linewidths=0.5)
            axes[-1].imshow(sitk.GetArrayFromImage(label[:, :, 40]), cmap='Reds', vmin=0, vmax=1.5, alpha=0.6)
            axes[-1].contour(sitk.GetArrayFromImage(label[:, :, 40]), colors='black', linewidths=0.5)
            axes[i].set_title(f"Label {i+1}")
            axes[i].axis('off')
        axes[-2].imshow(sitk.GetArrayFromImage(fused_result[:, :, 40]), cmap='Blues', vmin=0, vmax=1)
        axes[-2].contour(sitk.GetArrayFromImage(fused_result[:, :, 40]), colors='black', linewidths=0.5)
        axes[-2].set_title("Fused Label")
        axes[-2].axis('off')
        axes[-1].imshow(sitk.GetArrayFromImage(fused_result[:, :, 40]), cmap='Blues', vmin=0, vmax=1.5, alpha=0.8)
        axes[-1].contour(sitk.GetArrayFromImage(fused_result[:, :, 40]), colors='black', linewidths=0.5)
        axes[-1].set_title("Combination plot")
        axes[-1].axis('off')
        plt.show()
        
    def fusion_experiment(self, fusion_methods=["STAPLE", "MayorityVoting", "SIMPLE"], max_atlases=11, similar=["most", "least"]):
        """Run experiment to determine fusion method and number of atlases used for best results
        More efficient to run this code than the normal function for every setting"""
        
        # Select N atlas images to register onto target image based on NMI score
        atlas_indexes_full = list(range(len(self.data_atlas_paths)))
        atlas_indexes = self.atlas_selection(atlas_indexes_full, max_atlases)

        # List that will contain all the deformed labels to fuse
        labels_to_fuse = []
        
        target_name = self.data_target_path[0].split("\\")[-2]

        # Perform registration for each randomly selected atlas image
        for atlas_index in atlas_indexes:
            atlas_name = self.data_atlas_paths[atlas_index][0].split("\\")[-2]
            registration_name = f"Target-{target_name}_Atlas-{atlas_name}"

            registration = Registration(atlas_image_path=self.data_atlas_paths[atlas_index][0],
                                        atlas_label_path=self.data_atlas_paths[atlas_index][1],
                                        target_image_path=self.data_target_path[0],
                                        target_label_path=self.data_target_path[1],
                                        registration_name=registration_name,
                                        parameter_file=self.parameter_file)

            registration.perform_registration(plot=self.plot)


            print(
                f"Computing metrics for deformed atlas label {atlas_name} with target label {target_name}")
            dice, hd, hd95, recall, fpr, fnr = compute_metrics(label1_path=registration.atlas_label_deformed_path,
                                                                label2_path=registration.target_label_path, plot=self.plot)

            results = pd.DataFrame([{"parameter_file": self.parameter_file,
                                        "target": target_name, "atlas": atlas_name,
                                        "dice": dice, "hd": hd, "hd95": hd95, "recall": recall,
                                        "fpr": fpr, "fnr": fnr}])

            self.validation_results = pd.concat(
                [self.validation_results, results], ignore_index=True)

            # Read the deformed labels one by one and store them in a list
            registered_label_image = sitk.ReadImage(
                registration.atlas_label_deformed_path, sitk.sitkUInt8)
            labels_to_fuse.append(registered_label_image)

        for fusion_method in fusion_methods:
            for nr_atlases in range(1, max_atlases+1):
                for similarity in similar:

                    if similarity=="most":
                        fused_result = fuse_images(
                            labels_to_fuse[:nr_atlases], method=fusion_method, class_list=[0,1])
                    elif similarity=="least":
                        fused_result = fuse_images(
                            labels_to_fuse[-nr_atlases:], method=fusion_method, class_list=[0, 1])
                    # else:
                    #     fused_result = fuse_images(labels_to_fuse[:nr_atlases], method=fusion_method)

                    # Write the fused result to output file
                    fused_atlas_label_path = f"results/Target-{target_name}_Fusion_{fusion_method}_NrOfAtlases_{nr_atlases}_FusedLabel.mhd"
                    sitk.WriteImage(fused_result, fused_atlas_label_path)

                    # Plot individual labels and fused label
                    if (self.plot == True) or (self.plot == "final"):
                        self.plot_fusion(labels_to_fuse, fused_result)


                    print(
                        f"Computing metrics for the fused atlas label with target label {target_name}")
                    fused_dice, hd, hd95, recall, fpr, fnr = compute_metrics(label1_path=fused_atlas_label_path,
                                                                            label2_path=registration.target_label_path, plot=self.plot)

                    results = pd.DataFrame([{"parameter_file": self.parameter_file, "fusion_method": fusion_method, "NrOfAtlases": nr_atlases,
                                            "target": target_name, "atlas": "fused_atlas", "dice": fused_dice, "similarity": similarity,
                                            "hd": hd, "hd95": hd95, "recall": recall, "fpr": fpr, "fnr": fnr}])

                    self.validation_results = pd.concat(
                        [self.validation_results, results], ignore_index=True)
            

        return self.validation_results
        

# TODO could move these functions to a utils file
def img_from_path(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def overlay_from_segmentation(img_segmentation):
    return np.ma.masked_where(img_segmentation == 0, img_segmentation)
    
def plot_dice(label1_img, label2_img, pixel_cutoff=50):
    """Plot the overlap between two labels along with their individual regions.
    
    Args:
        label1_img (ndarray): First label/segmentation image.
        label2_img (ndarray): Second label/segmentation image.
    """
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(label1_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                  cmap='Reds', vmin=0, vmax=1)
    ax[0].contour(label1_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                   colors='black', linewidths=0.5)
    ax[0].set_title("Deformed atlas label")
    ax[0].axis('off')

    ax[1].imshow(label2_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                  cmap='Blues', vmin=0, vmax=1)
    ax[1].contour(label2_img[40, pixel_cutoff:-pixel_cutoff, pixel_cutoff:-pixel_cutoff],
                   colors='black', linewidths=0.5)
    ax[1].set_title("Target label")
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

    plt.tight_layout(w_pad=-20)
    plt.show()

def compute_metrics(label1_path, label2_path, plot=False):
    """From two images, compute dice, hd, hd95, recall, fpr, fnr"""
    label1_img = img_from_path(label1_path)
    label2_img = img_from_path(label2_path)
    
    if plot:
        plot_dice(label1_img, label2_img)
    
    # Compute metrics with seg_metrics library
    metrics = sg.write_metrics(labels=[0,1],
                                gdth_img=label1_img,
                                pred_img=label2_img,
                                TPTNFPFN=False,
                                spacing=[0.488281, 0.488281, 1],
                                metrics=['hd', 'hd95','dice','recall','fpr','fnr'])
    metrics_dict = metrics[0]

    dice = metrics_dict['dice'][1]
    hd=metrics_dict['hd'][1]
    hd95 = metrics_dict['hd95'][1]
    recall = metrics_dict['recall'][1]
    fpr = metrics_dict['fpr'][1]
    fnr = metrics_dict['fnr'][1]
        
    return dice, hd, hd95, recall, fpr, fnr

