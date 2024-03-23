"""
Atlas-based segmentation prostate

8DM20 - Group 6
O. Capmany Dalmas, P. Del Popolo, Z. Farshidrokh, D. Le, J. van der Pas, M. Vroemen
Utrecht University & University of Technology Eindhoven

Plan:
- Test registration parameters and choose one (vary parameters, e.g. resolution, bending penalty, etc.)
- Test fusion methods and choose one
- Atlas selection (how similar are patients with NMI)

Preprocessing: determine prostate volumes etc?

Validation study:
- Total 15 patients (10 for tuning, 5 for validation)
- Registration parameter tuning on 10 patients 
  - Metric, transform, optimizer, resolutions, bending penalty, etc. (ROI?)
  - Methods from papers
  - Outside the box ideas
- Atlas fusion on 10 patients
  - Fusion methods (Majority voting, STAPLE, etc.)
  - Atlas selection (pairs with highest NMI)

Questions:
- How to use the 15 patients in determining best parameters?

"""

#%%
import pandas as pd
from dataset import Dataset
from registration import *

dataset = Dataset()

# %%%%%%%%%%%%%% Basic registration %%%%%%%%%%%%%%%
def test_registration(parameter_index=1, plot=True):
    """Function to test registration functionality
    
    For multisep registration, input a list of parameter files starting with Par0001translation.txt 
    as doing e.g. first rigid and then bspline registration will not work.
    
    Klein et. al. 2010 reports to find the best registration by using the the parameters in parameter_index=0
    
    Par0043rigid does not give a binary label as output and isrigid and therefore not that accurate 
    and Par0055 gives an error
    
    """
    parameter_files = [
        ["Par0001translation.txt", "Par0001bspline64.txt", "Par0001bspline32.txt",
            "Par0001bspline16.txt", "Par0001bspline08.txt"],
        ["Par0001translation.txt", "Par0001bspline64.txt"],
        ["Par0001translation.txt", "Par0001bspline16.txt"],
        ["Par0001translation.txt", "Par0001bspline4.txt"],
        "Par0001bspline16.txt",
        ["Par0001translation.txt", "Par0001affine.txt"],
        "Par0043rigid.txt",
        # RuntimeError: D:\a\im\_skbuild\win-amd64-3.10\cmake-build\_deps\elx-src\Core\Main\itkElastixRegistrationMethod.hxx:380: ITK ERROR: ElastixRegistrationMethod(000002153ED08070): Internal elastix error: See elastix log (use LogToConsoleOn() or LogToFileOn()).
        "Par0055.txt"
    ]
    results = pd.DataFrame()
    for j in range(0,5):
        target_index = j
        atlas_index = j+5
        registration = Registration(atlas_image_path=dataset.data_paths[atlas_index][0],
                                    atlas_label_path=dataset.data_paths[atlas_index][1],
                                    target_image_path=dataset.data_paths[target_index][0],
                                    target_label_path=dataset.data_paths[target_index][1],
                                    registration_name=f"Param{parameter_index}_T{target_index}_A{atlas_index}",
                                    parameter_file=parameter_files[parameter_index])

        registration.perform_registration(plot=plot)

        # Validate
        dice = calculate_dsc(label1_path=registration.atlas_label_deformed_path,
                            label2_path=registration.target_label_path, plot=plot)
        df_dice = pd.DataFrame([{"parameter_file": parameter_files[parameter_index],
                                "target_index": target_index, "atlas_index": atlas_index, "dice": dice}])
        results = pd.concat([df_dice, results], ignore_index=True)

    return results

# results = test_registration(parameter_index=0)


# %%%%%%%%%%%%%% Multi-atlas registration %%%%%%%%%%%%%%%
# Goal: determine best fusion method and nr_atlas_registrations
parameter_file = ["Par0001translation.txt", "Par0001bspline64.txt"]  # To be determined
fusion_methods = ["STAPLE", "majorityvoting", "itkvoting", "SIMPLE"]

def test_multi_registration(fusion_index=0):
    """Function to test multi-atlas registration functionality"""

    # Perform multi-atlas registration on ONE target image
    multi_registration_fusion = MultiRegistrationFusion(
        dataset=dataset, parameter_file=parameter_file, fusion_method=fusion_methods[fusion_index])
    fused_atlas_label_path, results = multi_registration_fusion.perform_multi_atlas_registration(target_index=2, nr_atlas_registrations=2)

    return results

results = test_multi_registration(fusion_index=3)
print(results)
