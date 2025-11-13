#!/bin/bash

data=data/dimer_cylinder_train_val_test/
python wgan_gp_main.py -a fc -v cross -e 3 -lp True -em False -ddir ${data}