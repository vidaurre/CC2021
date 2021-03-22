The main analyses are run through the scripts: 

- run_TUDA_Fig3, which has to do with between-trial temporal differences
- get_TGMs_Fig4and5, which has to do with standard decoding and temporal generalisation matrices (TGMs)

Further, preprocess_data.m contains the code to put the data in the format used in the rest of the code, 
as well as code for separating the signal into the oscillatory and non-oscillatory parts.  
 
The directory DecodingMethods contains additional scripts used for standard decoding.
The directory PermutationTesting contains scripts related to permutation testing. 
The director extra_TUDAfunctions contains advanced functions related to TUDA. 

The code require the HMM-MAR toolbox, which contains the core TUDA code: 
https://github.com/OHBA-analysis/HMM-MAR

The data used here was made public by Radek Cichy, and can be found here: 
http://userpage.fu-berlin.de/rmcichy/fusion_project_page/main.html

A detailed description of the paradigm can be found in:
Cichy RM, Pantazis D, Oliva A (2016). Similarity-based fusion of MEG and fMRI reveals spatio-temporal dynamics in human cortex during visual object recognition. Cerebral Cortex 26, 3563-3579





