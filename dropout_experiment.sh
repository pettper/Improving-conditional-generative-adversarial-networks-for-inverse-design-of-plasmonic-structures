#!/bin/bash

python wgan_gp_main.py --settings settings/dropout_experiment/fcgan_no_dropout.yaml
python wgan_gp_main.py --settings settings/dropout_experiment/fcgan_with_dropout.yaml
python wgan_gp_main.py --settings settings/dropout_experiment/dcgan_no_dropout.yaml
python wgan_gp_main.py --settings settings/dropout_experiment/dcgan_with_dropout.yaml