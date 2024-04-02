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

import pandas as pd
from dataset import Dataset
from registration import *
import random

########## Basic registration ##########
# Goal: determine registration parameters and choose one
def registration_experiment(data_paths, parameter_files, experiment_name, plot=True):
    """Function to experiment registration to determine optimal parameter file"""

    results = pd.DataFrame()
    # Go over every image as target and choose n random atlas indexes
    for parameter_file in parameter_files:
        print(f"Registration with parameter file: {parameter_file}")
        try:
            for target_index in range(len(data_paths)):
                target_name = data_paths[target_index][0].split("\\")[-2]
                
                # Portion of code to select atlas images based on similarity
                atlas_indexes = [idx for idx in range(len(data_paths)) if idx != target_index]
                nmi_scores = []
                for atlas_index in atlas_indexes:
                    target_image = itk.imread(data_paths[target_index][0], itk.F)
                    atlas_image = itk.imread(data_paths[atlas_index][0], itk.F)
                    # Convert images to arrays, this datatype is required to calculate the NMI scores
                    target_array = itk.array_from_image(target_image)
                    atlas_array = itk.array_from_image(atlas_image)
                    nmi = normalized_mutual_information(target_array, atlas_array)
                    # List of tuples containing the atlas index and the obtained NMI score
                    nmi_scores.append((atlas_index, nmi))
                # Sort by NMI scores and select the first nr_atlas_registrations indices, corresponding to the highest NMI scores
                sorted_nmi_scores = sorted(nmi_scores, key=lambda x: x[1], reverse=True)
                atlas_indices = [idx for idx, _ in sorted_nmi_scores]
                # Select most similar, middle and least similar atlas image
                selected_atlas_indices = [atlas_indices[0], atlas_indices[len(atlas_indices)//2], atlas_indices[-1]]
                
                similarities = ["most", "middle", "least"]
                i = 0
                for atlas_index in selected_atlas_indices:
                    similarity = similarities[i]
                    atlas_name = data_paths[atlas_index][0].split("\\")[-2]
                    # Registration
                    registration = Registration(atlas_image_path=data_paths[atlas_index][0],
                                                atlas_label_path=data_paths[atlas_index][1],
                                                target_image_path=data_paths[target_index][0],
                                                target_label_path=data_paths[target_index][1],
                                                registration_name=f"Param{parameter_file}_T{target_index}_A{atlas_index}",
                                                parameter_file=parameter_file)

                    registration.perform_registration(plot=plot)

                    # Validate
                    print(f"Computing metrics for the fused atlas label {atlas_name} ({similarity} similar) with target label {target_name}")
                    dice, hd, hd95, recall, fpr, fnr = compute_metrics(label1_path=registration.atlas_label_deformed_path,
                                                                            label2_path=registration.target_label_path, plot=plot)

                    results_temp = pd.DataFrame([{"parameter_file": parameter_file, "target": target_name, "atlas": atlas_name, "similarity": similarity,
                                            "dice": dice, "hd": hd, "hd95": hd95, "recall": recall, "fpr": fpr, "fnr": fnr}])
                    i += 1
                    
                    results = pd.concat([results_temp, results], ignore_index=True)
                    # results.to_csv(f'result_tables/OptimizationResults_{experiment_name}_notfinished.csv', index=False)
        except:
            print(f"Error in registration with parameter file {parameter_file}")
            continue

    results.to_csv(f'result_tables/OptimizationResults_{experiment_name}.csv', index=False)
    
    return results

########## Multi-atlas registration ##########
# Goal: determine best fusion method and nr_atlas_registrations

def fusion_experiment(data_paths, parameter_file, fusion_methods, max_atlases, similar):
    """Function to experiment multi-atlas registration to determine optimal parameters"""
    results = pd.DataFrame()

    for data_target_path in data_paths:
        data_atlas_paths = [path for path in data_paths if path != data_target_path]
        
        multi_registration_fusion = MultiRegistrationFusion(data_atlas_paths=data_atlas_paths, 
                                                            data_target_path=data_target_path,
                                                            parameter_file=parameter_file,
                                                            fusion_method="STAPLE", plot=False)
    
        results_taregt = multi_registration_fusion.fusion_experiment(
            fusion_methods=fusion_methods, max_atlases=max_atlases, similar=similar)
        
        results = pd.concat([results_taregt, results], ignore_index=True)
    
    results.to_csv(f"result_tables/OptimizationResults_FusionMethods_{parameter_file}.csv", index=False)
            
    return results

# TODO: Also experiments where atlases are chosen at random?

########## Validate/Deploy model ##########
def deploy_model(data_paths_target, data_paths_atlas, validate=False, fusion_method="SIMPLE", similar="most",
                 nr_atlas_registration=6, parameter_file=["Par0001translation.txt", "Par0001bspline16.txt"]):
    
    result_metrics = pd.DataFrame()
    result_paths = []
    for data_target_path in data_paths_target:
        multi_registration_fusion = MultiRegistrationFusion(data_atlas_paths=data_paths_atlas,
                                                            data_target_path=data_target_path,
                                                            parameter_file=parameter_file,
                                                            fusion_method=fusion_method, plot=True)
            
        fused_atlas_label_path = multi_registration_fusion.perform_multi_atlas_registration(
            nr_atlas_registrations=nr_atlas_registration, validate=validate, similar=similar)
        
        result_paths.append(fused_atlas_label_path)
        if validate:
            result_metrics = pd.concat([multi_registration_fusion.validation_results, result_metrics], ignore_index=True)
    
    if validate:    
        result_metrics.to_csv(
            f"result_tables/validation_results.csv", index=False)
        return result_metrics, result_paths
    
    return result_paths
    


if __name__ == '__main__':
    RUN = 3
        
    #### Data selection ####
    dataset_optimize = Dataset(dirname="data/data_optimize")
    dataset_validate = Dataset(dirname="data/data_validate")
    dataset_test = Dataset(dirname="data/data_test")
    
    #### Registration Parameters - Basic #####
    if RUN==0:
        parameter_files_simple = [
            "Par0001bspline16_1res.txt",
            "Par0001bspline16_8res.txt",
            "Par0001bspline16_500.txt",
            "Par0001bspline16_10000.txt",
            "Par0001bspline16_adaptiveGD.txt",
            "Par0001bspline16_CC.txt",
            "Par0001bspline16.txt",
            "Par0001bspline32.txt",
            "Par0001bspline64.txt",
            "Par0001rigid.txt",
            "Par0001translation_1res.txt",
            "Par0001translation_8res.txt",
            "Par0001translation_500.txt",
            "Par0001translation_10000.txt",
            "Par0001translation_adaptiveGD.txt",
            "Par0001translation_CC.txt",
            "Par0001translation.txt",
            "Par0001affine_1res.txt",
            "Par0001affine_8res.txt",
            "Par0001affine_500.txt",
            "Par0001affine_10000e.txt",
            "Par0001affine_adaptiveGD.txt",
            "Par0001affine_CC.txt",
            "Par0001affine.txt",
            "Par0001bspline04.txt",
            "Par0001bspline08.txt",
            "Par0043rigid.txt",
            "Par0055.txt"	
        ]
        results_simple_parameters = registration_experiment(data_paths=dataset_optimize.data_paths,
                                                            parameter_files=parameter_files_simple,
                                                            experiment_name="simple_parameters",
                                                            plot=True)

    #### Registration Parameters - MultiStep #####
    if RUN==1 :
        parameter_files_multistep = [
            ["Par0001translation.txt", "Par0001affine.txt"], ["Par0001translation_CC.txt", "Par0001affine_CC.txt"],
            ["Par0001translation.txt", "Par0001bspline16.txt"], ["Par0001translation_CC.txt", "Par0001bspline16_CC.txt"], 
            ["Par0001affine.txt", "Par0001bspline16.txt"], ["Par0001affine_CC.txt", "Par0001bspline16_CC.txt"],
            ["Par0001translation.txt", "Par0001affine.txt", "Par0001bspline16.txt"], ["Par0001translation_CC.txt", "Par0001affine_CC.txt", "Par0001bspline16_CC.txt"], 
                                     ]
        results_multistep_parameters = registration_experiment(data_paths=dataset_optimize.data_paths,
                                                            parameter_files=parameter_files_multistep,
                                                            experiment_name="multistep_parameters",
                                                            plot=False)

    #### Multi-Atlas Regsitration Prameters #####
    if RUN==2:
        result_fusion = fusion_experiment(data_paths=dataset_optimize.data_paths, similar=["most", "least"],
                                          parameter_file=["Par0001translation.txt", "Par0001bspline16.txt"],
                                          fusion_methods=["itkvoting", "SIMPLE", "STAPLE"], max_atlases=11)
        result_fusion = fusion_experiment(data_paths=dataset_optimize.data_paths, similar=["most", "least"],
                                          parameter_file=["Par0001translation_CC.txt", "Par0001bspline16_CC.txt"],
                                          fusion_methods=["itkvoting", "SIMPLE", "STAPLE"], max_atlases=11)

    #### Validation with known labels #####
    if RUN==3:
        result_val, result_paths_val = deploy_model(data_paths_target=dataset_validate.data_paths, data_paths_atlas=dataset_optimize.data_paths,
                                                    validate=True, fusion_method="SIMPLE", nr_atlas_registration=10, similar="least",
                                            parameter_file=["Par0001translation.txt", "Par0001bspline16.txt"])
    
    #### Model Deployment on unlabeled test data #####
    if RUN==4:
        result_paths_test = deploy_model(data_paths_target=dataset_test.data_paths, data_paths_atlas=dataset_optimize.data_paths,
                                            validate=False, fusion_method="SIMPLE", nr_atlas_registration=10, 
                                            parameter_file=["Par0001translation.txt", "Par0001bspline16.txt"])
    
    
