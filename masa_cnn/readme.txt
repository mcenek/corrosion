Masa's Convolution Network For Corrosion Detection

The trained model can be found in the cnn directory. The analysis_module has useful visualization tools for in-depth analysis on images and any extracted features.

FILES:
cnn/
analysis_modle/

USAGES:

A. For training and testing corrosion images
    1) navigate to cnn folder
    2) for training run cnn.py with the train option. This will only work if you have the Corrosion Dataset in the home directory named as Corrosion_Dataset

        python cnn.py train

    3) for testing a corrosion image with the trained model, run cnn.py with the test option. DON'T USE BRACKETS!!!!

        python cnn.py test [path to image to be tested] [path to tensorflow model ckpt]

    4) for visualizing the training on tensorboard execute runtensorboard.sh

        ./runtensorboard.sh

B. For post processing segmented images of the model testing

    1) navigate to home directory of this masa's project
    2) execute smooth.py with

