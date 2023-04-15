# InstanceSearch
Instance search assignment for CS4186 Computer Vision & Image Process

## Experiment reproduction
### Best method: SIFT
* Customized file path and parameters (explained in report) in utils.params.py
    * DATASET_PATH = project working directory (root directory)
    * QUERY_DIR = query image directory
    * DATA_DIR = dataset directory
    * GT_DIR = ground truth directory
    * DES_DIR = directory for descriptor files
    * COL_DIR = directory for color files
    * RET_DIR = SIFT result directory
    * LBP_RET_DIR = LBP result directory
    * LBP_DIR = directory LBP feature files
* Generate descriptor files for image i(i = 0, 1, ..., 5019)
    * Interface: from utils.fileIO import generate_descriptor_file
* Retrieve top-10 Match
    * Interface: from SIFT.Interface.match import match_main
* Ranklist
    * Interface: rank_list()

**Last but not least, if you have problems in generating descriptor files (do not know how / extremely long running time), please feel free to reach out to me for descriptor files generated in my experiment. Unfortunately, I cannot upload zip file of them on canvas due to huge file size...**

**Email: xyli45-c@my.cityu.edu.hk**