#!/bin/bash

data=data/dimer_cylinder_train_val_test/
python cnn_main.py -a effv2 -v cross -e 11 -ddir ${data}
