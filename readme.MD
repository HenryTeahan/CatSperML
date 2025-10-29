# Guide
This guide will explain how to run a demonstration of the model.
data/toy_indoles_aligned.sdf contains some randomly generated substituted indoles produced in random_indoles.ipynb. A random selection of 10 of these indoles are given an active label, and any molecules in the dataset with a Tanimoto similarity of >0.7 (using ECFP4) are also given an active label. This gives the model a chance to familiarize itself to some patterns. 
\\
The result of this is an .sdf file, toy_indoles.sdf, where each molecule has a mol.prop of its binary activity for classification.
\\
The workflow can be run as follows:
1. python align.py --sdf "path_to_toy_indoles_file.sdf" -> This creates a new sdf (path_to_toy_indoles_file_aligned.sdf), with indoles aligned using the alignment protocol in the paper.
2. python parameter_optimization.py --sdf "path_to_toy_indoles_file_aligned.sdf" ... You can also choose to alter the hyperparameter optimization ranges, for more info run: python parameter_optimization.py -h
3. python train.py --sdf "path_to_toy_indoles_file_aligned.sdf" --load_opt_params (MUST USE IF YOU WANT TO USE YOUR OPTIMIZED HYPERPARAMETERS FROM 2.)
4. python screen.py --sdf_t "path_to_toy_indoles_file_aligned.sdf" --sdf_s "path_to_screening_library" Note: You must have aligned you screening and training data to the same indole backbone template. I recommend running python align.py on the screening dataset if in doubt.
5. Success.

You can now go and view your hits using processing_hits.ipynb! Here you can grow familiar to the various data-processing tools developed.