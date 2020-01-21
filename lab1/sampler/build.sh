#!/bin/bash
gcc -o adc_sampler  adc_sampler.c -lpigpio  -lpthread  -lm
