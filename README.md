# 8DM20 - CSMIA - Atlas-based segmentation of prostate

## Group 6

* Olga Capmany Dalmas
* Paula Del Popolo
* Zahra Farshidrokh
* Daniel Le
* Jelle van der Pas
* Marcus Vroemen

## Project 
This repository contains the code for performing atlas-based segmentation of the prostate... 

## Requirements

* Python == 3.x
* Elastix == 5.1.0
* elastix-py

## Usage
WIP

## Folder Structure
```
ProstateRegistrationG6/
├───main.py - main script
├───dataset.py - class for dataset
│
├───data/ - folder containing data files 
|   └── pXXX/ - contains image and segmentation mask data in .mhd and .zraw format
│
├───parameters/ - folder containing parameter files
│   └── parameters.txt - holds configuration for registration
|
└───saved/ - generated log files
```

## Experiments & More
1. Data split: 12 for optimizing and 3 for final validation
2. Registration optimization
    1. From 12 patients select 1 target (fixed) and 3 atlas (moving)
        - Choose more atlas? 
        - Choose randomly, structured (T1 with A2,3,4 and T2 with A3,4,5), or based on similarity?
    2. Registration between target and atlas with specific parameters
    3. Compute dice between deformed atlas label and target's real label and average Dice
    4. Repeat for all 12 patients and make final average of dice
        - Doing 12*3 registrations per parameter set is costly. Either make this less or not choose too many parameter options
        - Could for example do an exploratorive parameter tuning approach just to know what would definitely not work
3. Atlas fusion
    1. From 3 validation patients, choose 1 target and 3 atlas images (from set of 12)
        - Could do more atlas images
        - Choose 3 atlas images based on similarity with targer?
    2. Registration with optimized parameters to get deformed atlas labels
    3. Fuse atlas labels into one with one of the methods
        - STAPLE, VOTING, SIMPLE, majorityvoting
    4. Repeat for 3 validation images to determine the best average dice
4. Testing
    - With optimal registration parameters, atlas fusion technique and other choises, perform segmentation on unlabeled images and determine test score (send in results to Josien)

### Data split
The available data contains 15 patients with expert annotated segmentations in the form of binary label masks. The data is split where 12 patients are selected for optimization and 3 for validating the model (nested CV?). The optimization include determining the optimal registration method and parameters, and the atlas label fusion method.

### Registration parameters
The registration algorithm has to be optimized by determining the best registration method and parameters. Pairs of target (fixed) and atlas (moving) images are selected form the 12 patient data split and the method with the highest mean dice score between deformed atlas label and target label is selected for further experiments. To be more specific, every image is selected once as target image, then 3 atlas images with the highest similarity to the target image are selected and registered to the target space, and the dice between the deformed labels and the target image label are averaged. This is done for every of the 12 images.
 

The Elastix model zoo contains a large collection of parameter files for registration using elastix, including files for 3D prostate MRI (https://elastix.lumc.nl/modelzoo/). The following files are used in initial registration experiments: 
- Par0001 (interpatient; affine + B-spline transformation; mutual information) [1]
- Par0043 (intra-subject; multi-resolution, rigid, Mutual Information metric (Mattes) with Adaptive Stochastic Gradient Descent optimizer) [2] 
- Par0055 (intra-subject cine-MRI motion; rigid transformation) [3] 

### Combining atlas segmentations
We use the LabelFusion python library to combine the atlas segmentations into one label
Available LabelFusion:
Voting (ITK): DOI:10.1016/j.patrec.2005.03.017
STAPLE (ITK): DOI:10.1109/TMI.2004.830803
Majority Voting: DOI:10.1007/978-3-319-20801-5_11
SIMPLE: DOI:10.1109/tmi.2010.2057442

## Bibliography
[1] S. Klein, U.A. van der Heide, I.M. Lips, M. van Vulpen, M. Staring and J.P.W. Pluim, "Automatic Segmentation of the Prostate in 3D MR Images by Atlas Matching using Localised Mutual Information," Medical Physics, vol. 35, no. 4, pp. 1407 - 1417, April 2008

[2] Maspero M, Savenije MH, Dinkla AM, Seevinck PR, Intven MP, Jurgenliemk-Schulz IM, Kerkmeijer LG, van den Berg CA. Dose evaluation of fast synthetic-CT generation using a generative adversarial network for general pelvis MR-only radiotherapy. Physics in Medicine & Biology. 2018 Sep 10;63(18):185001. doi:https://doi.org/10.1088/1361-6560/aada6d;

[3] de Muinck Keizer DM, Kerkmeijer LG, Maspero M, Andreychenko A, van Zyp JV, Van den Berg CA, Raaymakers BW, Lagendijk JJ, de Boer JC. Soft-tissue prostate intrafraction motion tracking in 3D cine-MR for MR-guided radiotherapy. Physics in Medicine & Biology. 2019 Dec 5;64(23):235008. doi:https://doi.org/10.1088/1361-6560/ab5539.