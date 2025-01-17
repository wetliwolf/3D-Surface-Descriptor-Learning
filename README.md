# 3D-Surface-Descriptor-Learning

This code represents a deep learning method for local descriptors of 3D surface shapes. The following paper describes this method in detail:

      ------------------------------------------------------------
      Hanyu Wang, Jianwei Guo, Dong-Ming Yan, Weize Quan, Xiaopeng Zhang. 
      Learning 3D Keypoint Descriptors for Non-Rigid Shape Matching. 
      ECCV 2018.
      ------------------------------------------------------------

An advanced variation of this method is available in [LPS](https://github.com/yiqun-wang/LPS). Please consider citing the paper mentioned above if you utilize the code or its components.

# License

This software is freely available under the GNU General Public License published by the Free Software Foundation. You are free to distribute and/or modify this code according to either version 2 of the License, or, at your disposition, any later version.

# Instructions

Three folders exist in this repository. The 'cpp' folder uses 'matlab' for Geometry Image (GI) creation while the 'python' folder is for network training and testing. Details of the usage is present below:

1. Compile matlab project using command -- 'MCC matlab "mcc -W cpplib:libcompcur -T link:lib compute_curvature.m"' to get 'libcompcur.dll' which should be added to the CPP project

2. Build cpp solution: to generate geometry images. Modify CMakeLists and Build solution accordingly. Run 'GIGen.exe config.ini' to generate GI.

3. Python project works for network training and testing. Copy the geometry images from previous step into the server.

-Train network:

    Run scripts in the following sequence:
    	'classify_gi_by_pidx_and_split.py'
    	'tfr_gen.py'
    	'train_softmax256.py'
    	'train_mincv_perloss.py'

-Test to generate descriptor:

    Run 'descGen.py' to generate descriptor using test dataset.

# Contact

For any questions, comments, or suggestions, feel free to reach out at: wetliwolf@gmail.com

October, 2018

Copyright (C) 2018 