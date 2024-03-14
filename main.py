"""
Atlas-based segmentation prostate

8DM20 - Group 6
O. Capmany Dalmas, P. Del Popolo, Z. Farshidrokh, D. Le, J. van der Pas, M. Vroemen
Utrecht University & University of Technology Eindhoven

"""
#%%
import pandas as pd
from dataset import Dataset
from registration import *

dataset = Dataset()

# TODO test currently available parameter files
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
    "Par0043rigid.txt"
]

#%% Perform multi-atlas registration on ONE target image
multi_registration_fusion = MultiRegistrationFusion(dataset, parameter_files[0], fusion_method="MajorityVoting")
fused_atlas_label_path, results = multi_registration_fusion.perform_multi_atlas_registration(target_index=3, nr_atlas_registrations=3)

#%%
for parameter_file in parameter_files[1:]:
    print(parameter_file)
    multi_registration_fusion = MultiRegistrationFusion(dataset, parameter_file, fusion_method="MajorityVoting", validation_results= results)
    fused_atlas_label_path, results = multi_registration_fusion.perform_multi_atlas_registration(target_index=3, nr_atlas_registrations=3)
    
# #%% Perform multi-atlas registration on ALL target image
# multi_registration_fusion = MultiRegistrationFusion(dataset, parameter_files[0], fusion_method="MajorityVoting")
# results = multi_registration_fusion.full_multi_atlas_registration(nr_atlas_registrations=3)

# # %% Perform multi-atlas registration on ALL target images with ALL parameter files
# results = pd.DataFrame(columns=["parameter_file", "fusion_method", "target_index", "atlas_index", "dice"])

# for parameter_file in parameter_files:
#     multi_registration_fusion = MultiRegistrationFusion(dataset, parameter_file, 
#                                                         fusion_method="MajorityVoting", results=results)
#     results = multi_registration_fusion.full_multi_atlas_registration(nr_atlas_registrations=3)
    

#print(results)

#%%
filtered_results = results[results['atlas_index'] == 'fused_atlas']
print(filtered_results)