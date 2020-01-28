#!/bin/bash

sudo ./sampler/adc_sampler 1000
git add data/adcData.bin
git commit -m "New Lab2 Sample"
