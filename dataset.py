"""
Atlas-based segmentation prostate

8DM20 - Group 6
O. Capmany Dalmas, P. Del Popolo, Z. Farshidrokh, D. Le, J. van der Pas, M. Vroemen
Utrecht University & University of Technology Eindhoven

"""

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import glob

class Dataset():
    def __init__(self):
        self.dir = "data"
        self.data_paths = self.get_data_paths()

    def get_data_paths(self):
        """
        Returns all file paths of image and mask from project root directory.

        Returns:
            list: list of [image, mask] paths.
        """        
        filelist = glob.glob(os.path.join(self.dir, "*"))
        data_paths = list()

        for data_dir in filelist:
            image_mhd_path = os.path.join(data_dir, "mr_bffe.mhd")
            mask_mhd_path = os.path.join(data_dir, "prostaat.mhd")
            data_paths.append([image_mhd_path, mask_mhd_path])
        
        return data_paths
    
    def __length__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        """Returns array data from image and mask for a given index.

        Args:
            index (int): index file in dataset

        Returns:
            list: list of [image, mask] numpy.ndarray 
        """        
        data_path = self.data_paths[index]
        output = list()

        for i, path in enumerate(data_path):
            itk_image = sitk.ReadImage(path)
            itk_array = sitk.GetArrayFromImage(itk_image)
            output.append(itk_array)

        return output
    
    def plot(self, index, slice=40):
        """Plots image, mask and overlay for a given file index.

        Args:
            index (int): index data file
            slice (int, optional): index slice. Defaults to 40.
        """        
        [image, mask] = self.__getitem__(index)
        overlay = np.ma.masked_where(mask == 0, mask)
    
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image[slice,:,:], cmap="gray")
        axs[1].imshow(mask[slice,:,:], cmap="gray")
        axs[2].imshow(image[slice,:,:], cmap="gray")
        axs[2].imshow(overlay[slice,:,:], cmap="jet", alpha=0.6)
        titles = ["Image", "Mask", "Overlay"]

        for ax, title in zip(axs, titles):
            ax.set_title(title)
            ax.axis("off")
        fig.suptitle(f"Prostate image #{index}", y=0.85, fontsize=14)

        plt.tight_layout()
        plt.show()

