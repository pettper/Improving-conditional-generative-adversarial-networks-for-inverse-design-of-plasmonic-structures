#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.

python tests/performance_tests/wgan_gp_runtime_test.py -s settings/benchmarking/dcgan.yaml