#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

data=data/anisotropic_au_structures_train_val_test/
python tests/performance_tests/wgan_gp_runtime_test.py -a dc -v cross -e 15 -lp False -em False -ddir ${data} -dsz 3000 -b 64 -tb True