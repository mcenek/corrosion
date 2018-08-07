#!/bin/bash

dir='../Corrosion_Dataset/IRtest/defects/'
model='model/cnn_model.ckpt'
for f in $(ls $dir)
do
    python cnn.py test $dir$f $model
done
