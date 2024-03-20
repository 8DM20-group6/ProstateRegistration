"""
Atlas-based segmentation prostate

8DM20 - Group 6
O. Capmany Dalmas, P. Del Popolo, Z. Farshidrokh, D. Le, J. van der Pas, M. Vroemen
Utrecht University & University of Technology Eindhoven

Plan:
- Test registration parameters and choose one (vary parameters, e.g. resolution, bending penalty, etc.)
- Test fusion methods and choose one
- Atlas selection (how similar are patients with NMI)

"""

#%%
import pandas as pd
from dataset import Dataset
from registration import *

dataset = Dataset(n_validation=3)

# %%%%%%%%%%%%%% Basic registration %%%%%%%%%%%%%%%
# Goal: determine registration parameters and choose one
def registration_experiment(data_paths, parameter_files, n_atlas_registrations=1, plot=True):
    """Function to test registration functionality
    
    For multisep registration, input a list of parameter files starting with Par0001translation.txt 
    as doing e.g. first rigid and then bspline registration will not work.    
    """

    results = pd.DataFrame()
    for parameter_file in parameter_files:
        # Go over every image as target and choose n random atlas indexes
        for target_index in range(len(data_paths)):
            atlas_indexes = atlas_indexes = random.sample([idx for idx in range(
                len(data_paths)) if idx != target_index], n_atlas_registrations)
            for atlas_index in atlas_indexes:
                # Registration
                registration = Registration(atlas_image_path=data_paths[atlas_index][0],
                                            atlas_label_path=data_paths[atlas_index][1],
                                            target_image_path=data_paths[target_index][0],
                                            target_label_path=data_paths[target_index][1],
                                            registration_name=f"Param{parameter_file}_T{target_index}_A{atlas_index}",
                                            parameter_file=parameter_file)

                registration.perform_registration(plot=plot)

                # Validate
                dice = calculate_dsc(label1_path=registration.atlas_label_deformed_path,
                                    label2_path=registration.target_label_path, plot=plot)
                df_dice = pd.DataFrame([{"parameter_file": parameter_file,
                                        "target_index": target_index, "atlas_index": atlas_index, "dice": dice}])
                results = pd.concat([df_dice, results], ignore_index=True)

    return results

#%% 
# TODO: Can also try other modifications to the parameter files (change metric, resolution, etc)
#%%
parameter_files_simple = [
    "Par0001translation.txt", "Par0001rigid.txt", "Par0001affine.txt",
    "Par0001bspline64.txt", "Par0001bspline16.txt", "Par0001bspline04.txt"
]
results_simple_parameters = registration_experiment(
    data_paths=dataset.data_paths_optimize, parameter_files=parameter_files_simple, plot=True, n_atlas_registrations=1)
results_simple_parameters.to_csv('results_simple_parameters.csv', index=False)
#%%
parameter_files_multistep = [
    ["Par0001translation.txt", "Par0001affine.txt"], ["Par0001translation.txt", "Par0001bspline4.txt"],
    ["Par0001translation.txt", "Par0001bspline16.txt"], ["Par0001translation.txt", "Par0001bspline64.txt"],
    ["Par0001translation.txt", "Par0001bspline64.txt", "Par0001bspline32.txt", "Par0001bspline16.txt", "Par0001bspline08.txt"],
] 
results_multistep_parameters = registration_experiment(
    data_paths=dataset.data_paths_optimize, parameter_files=parameter_files_multistep, plot=True, n_atlas_registrations=1)
results_multistep_parameters.to_csv(
    'results_multistep_parameters.csv', index=False)

#%%%%%%%%%%%%%% Multi-atlas registration %%%%%%%%%%%%%%%
# Goal: determine best fusion method
# TODO: implement similarity metric for atlas selection (for every target, select 3 or 5 atlas images based on similarity)
parameter_file = ["Par0001translation.txt",
                  "Par0001bspline64.txt"]  # To be determined
fusion_methods = ["staple", "majorityvoting", "itkvoting", "SIMPLE"]

def test_multi_registration(parameter_index=0, fusion_index=0):
    """Function to test multi-atlas registration functionality"""

    # Perform multi-atlas registration on ONE target image
    multi_registration_fusion = MultiRegistrationFusion(
        dataset=dataset, parameter_file=parameter_file, fusion_method=fusion_methods[0])
    fused_atlas_label_path, results = multi_registration_fusion.perform_multi_atlas_registration(target_index=2, nr_atlas_registrations=2)

    # # Perform multi-atlas registration on ALL target images
    # multi_registration_fusion = MultiRegistrationFusion(
    #     dataset=dataset, parameter_file=parameter_files[0], fusion_method=fusion_methods[0])
    # results = multi_registration_fusion.full_multi_atlas_registration(nr_atlas_registrations=3)

    # # Perform multi-atlas registration on ALL target images with ALL parameter files
    # results = pd.DataFrame(columns=["parameter_file", "fusion_method", "target_index", "atlas_index", "dice"])

    # for parameter_file in parameter_files:
    #     multi_registration_fusion = MultiRegistrationFusion(dataset, parameter_file, 
    #                                                         fusion_method=fusion_methods[0], results=results)
    #     results = multi_registration_fusion.full_multi_atlas_registration(nr_atlas_registrations=3)
    

# test_multi_registration(parameter_index=0, fusion_index=1)

