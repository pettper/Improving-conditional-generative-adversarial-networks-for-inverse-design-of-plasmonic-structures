#!/bin/bash

rm -r tmp/*
python wgan_gp_main.py -s settings/wgan_gp_example.yaml
mv tmp/* delivery/