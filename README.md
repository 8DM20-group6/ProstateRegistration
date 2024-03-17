# 8DM20 - CSMIA - Atlas-based segmentation of prostate

## Group 6

* Olga Capmany Dalmas
* Paula Del Popolo
* Zahra Farshidrokh
* Daniel Le
* Jelle van der Pas
* Marcus Vroemen

## Project 
[Link to paper]()

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
### Registration parameters
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