Code for the paper "Improving conditional generative adversarial networks for
inverse design of plasmonic structures"

Petter Persson, petter.persson@umu.se



Description of repository

1. Root-folder contains main scripts.
2. data/ contains all data used in the project.
	a) dimer_cylinder_train_val_test and anisotropic_structures_train_val_test are split into training, validation, test sets. Use this data for training.
	b) Other folder contains raw data
	c) train_val_test_split.sh is a script that can be used to split a data set.
3. job_scripts/ contains the scripts used to train the models on HPC2N-Kebnekaise cluster.
4. src/ contains all source code.
5. tests/ contains all test code.
