#!/bin/bash 
wget -O ./model.h5 https://www.dropbox.com/s/79oajiwow0xzit9/model.h5?dl=1
python3 predict.py $1 $2
