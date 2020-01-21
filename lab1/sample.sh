#!/bin/bash

sudo ./sampler/adc_sampler 31250
git add data/adcData.bin
git commit -m "New Sample"
