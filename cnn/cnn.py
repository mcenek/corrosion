#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for TBI, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import cv2
import math
import sys
import os

#Python Modules
import constants
import featureReader

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

###################################################################

###################################################################
#1. Convolutional layer
#2. Pooling layers
#3. Convolutional layer
#4. pooling layer
#5. Fully connected layer
#6. Logits layer
###################################################################

####################################################################################################################################
#Helper Functions
####################################################################################################################################
def outputResults(image,mask,fout='segmentation.png'):
    #create the segmented image
    canvas = image.copy()
    canvas[mask == -1] = [0,0,0]
    canvas[mask == 0] = [0,0,0]
    canvas[mask == 1] = [255,255,255]

    #show the original image and the segmented image and then save the results
    cv2.imwrite(fout,canvas)

####################################################################################################################################
#######################################################################################
#######################################################################################
#Main Function

def main(unused_argv):

    #check the number of arguments given with running the program
    #must be at least two
    #argv[1] is the mode of operation {test,see,train}
    #argv[2] is the input image
    #argv[3] is the optional
    if not os.path.exists('log'):
        os.makedirs('log')

    if len(sys.argv) >= 2:

        #################################################################################################################
        #################################################################################################################
        #Define our Convolutionary Neural Network from scratch
        x1 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        x2 = tf.placeholder('float',[None,constants.IMG_SIZE,constants.IMG_SIZE,constants.IMG_DEPTH])
        y = tf.placeholder('float',[None,1])
        y1 = tf.placeholder('float',[None,1])
        y2 = tf.placeholder('float',[None,1])

        weights = {}
        biases = {}

        #magic number = width * height * n_convout
        magic_number = int((constants.CNN_LOCAL1 + constants.CNN_GLOBAL) * (constants.IMG_SIZE * constants.IMG_SIZE))

        #rust matter convolution network
        with tf.name_scope('model_rust'):

            with tf.name_scope('path1'):
                weights['w_rustmatter'] = tf.Variable(tf.random_normal([7,7,constants.IMG_DEPTH,constants.CNN_LOCAL1]))
                biases['b_rustmatter'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
                rust_conv1 = tf.nn.conv2d(x1,weights['w_rustmatter'],strides=[1,1,1,1],padding='SAME',name='rust_rust1')
                rust1 = tf.nn.relu(rust_conv1 + biases['b_rustmatter'])
                weights['w_rustmatter_p1b'] = tf.Variable(tf.random_normal([3,3,constants.CNN_LOCAL1,constants.CNN_LOCAL1]))
                biases['b_rustmatter_p1b'] = tf.Variable(tf.random_normal([constants.CNN_LOCAL1]))
                rust_conv2 = tf.nn.conv2d(rust1,weights['w_rustmatter_p1b'],strides=[1,1,1,1],padding='SAME',name='rust_rust2')
                rust2 = tf.nn.relu(rust_conv2 + biases['b_rustmatter_p1b'])
            with tf.name_scope('path2'):
                weights['w_rustmatter_global'] = tf.Variable(tf.random_normal([13,13,constants.IMG_DEPTH,constants.CNN_GLOBAL]))
                biases['b_rustmatter_global'] = tf.Variable(tf.random_normal([constants.CNN_GLOBAL]))
                grust_conv1 = tf.nn.conv2d(x1,weights['w_rustmatter_global'],strides=[1,1,1,1],padding='SAME',name='global_rust')
                rust3 = tf.nn.relu(grust_conv1 + biases['b_rustmatter_global'])
            with tf.name_scope('combined'):
                rust_activations = tf.concat([rust2,rust3],3)
                weights['out1'] = tf.Variable(tf.random_normal([magic_number,1]))
                biases['out1'] = tf.Variable(tf.random_normal([1]))
                output1 = tf.reshape(rust_activations,[-1,magic_number])
                predictions1 = tf.matmul(output1,weights['out1'])+biases['out1']
                out = tf.nn.sigmoid(predictions1)
                with tf.name_scope('cost'):
                    cost1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions1,labels=y1))
                    tf.summary.scalar('cost1',cost1)
                with tf.name_scope('optimizer'):
                    optimizer1= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost1)
                    op2list = [weights['out1'],biases['out1']]
                    optimizer2= tf.train.AdamOptimizer(constants.LEARNING_RATE).minimize(cost1,var_list=op2list)
                with tf.name_scope('accuracy'):
                    correct_prediction1 = tf.cast(tf.equal(tf.round(tf.nn.sigmoid(predictions1)),y1),tf.float32)
                    accuracy1 = tf.reduce_mean(correct_prediction1)
                    tf.summary.scalar('accuracy1',accuracy1)

        #################################################################################################################
        #################################################################################################################
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

####################################################################################################################################
####################################################################################################################################
####################################################################################################################################
#helper functions

        #training mode trained on the image
        if(sys.argv[1] == 'train'):
            #Run the session/CNN and train/record accuracies at given steps
            #net = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
            #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            with tf.Session() as sess:
                training_writer = tf.summary.FileWriter('./log',sess.graph)
                sess.run(init)

                #initialize log directory, model directory, and default accuracy
                acc = 0.00;
                modelpath = "model"
                logdir = 'log/traininglog.txt'
                if not os.path.exists(modelpath):
                    os.makedirs(modelpath)
                if not os.path.exists('log'):
                    os.makedirs('log')

                #train the model
                for epoch in range(constants.CNN_EPOCHS):
                    #get an image batch and train each model separately
                    batch_x,batch_y = featureReader.getPixelBatch(constants.BATCH_SIZE)
                    if epoch < 1000:
                        sess.run([optimizer1],feed_dict={x1: batch_x, y1: batch_y})
                    else:
                        sess.run([optimizer2],feed_dict={x1: batch_x, y1: batch_y})

                    #evaluate the model separately using a test set
                    if epoch % 1 == 0:

                        #merge summaries
                        merged = tf.summary.merge_all()
                        #evaluate test set
                        eval_x,eval_y = featureReader.getPixelBatch(constants.BATCH_SIZE)
                        summary, acc1 = sess.run([merged,accuracy1],feed_dict={x1:eval_x,y1:eval_y})
                        training_writer.add_summary(summary,epoch)

                        #save the model if it holds the highest accuracy or is tied for highest accuracy
                        if(acc1 >= acc):
                            acc = acc1
                            save_path = saver.save(sess,'model/cnn_model.ckpt')
                            print("highest accuracy found! model saved")

                        #print out to console and log
                        print('epoch: %i  rust: %.4f ' % (epoch,acc1))
                        with open(logdir,'a') as log_out:
                            log_out.write('epoch: %i   accuracy: %.4f  \n' % (epoch,acc1))


        #testing method needs a saved check point directory (model)
        elif(sys.argv[1] == 'test' and len(sys.argv) == 4):
            #get the directory of the checkpoint
            ckpt_dir = sys.argv[3]

            #read the image
            if os.path.isfile(sys.argv[2]):
                image = cv2.imread(sys.argv[2],cv2.IMREAD_COLOR)
                gray_img = cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)
                hogimg = featureReader.getHOG(image)
                wt = featureReader.extractWT(gray_img)
                image = np.concatenate((image,hogimg.reshape((hogimg.shape[0],hogimg.shape[1],1))),axis=-1)
                image = np.concatenate((image,wt.reshape((wt.shape[0],wt.shape[1],1))),axis=-1)

            #restore the graph and make the predictions and show the segmented image
            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess,ckpt_dir)
                print("session restored!")

                #we recreate the image by painting the best_guess mask on a blank canvas with the same shape as image
                #initialize counters and the height and width of the image being tested.
                #constants.IMG_SIZE is the img size the learned model uses for classifiying a pixel.
                h,w = image.shape[:2]
                count = 0
                count2 = 0
                best_guess = np.full((h,w),-1)
                raw_guess = np.full((h,w,constants.CLASSES),0.0).astype(np.float32)
                tmp = []
                i0 = int(constants.IMG_SIZE / 2)
                j0 = int(constants.IMG_SIZE / 2)

                #define our log file and pixel segmentation file name
                if not os.path.exists('results'):
                    os.mkdir('results')
                imgname = os.path.basename(sys.argv[2])
                modelname = os.path.dirname(sys.argv[3])
                logname = "results/rawoutput_" + str(os.path.splitext(os.path.basename(sys.argv[2]))[0]) + '_' + modelname + ".txt"
                seg_file = 'results/' + os.path.splitext(imgname)[0] + '_' + modelname + '_learnedseg' + ".png"

                #GO THROUGH EACH PIXEL WITHOUT THE EDGES SINCE WE NEED TO MAKE SURE EVERY PART OF THE PIXEL AREA
                #BEING SENT TO THE MODEL IS PART OF THE IMAGE
                for i in range(int(constants.IMG_SIZE / 2),int(len(image) - (constants.IMG_SIZE / 2))):
                    for j in range(int(constants.IMG_SIZE / 2),int(len(image[0]) - (constants.IMG_SIZE / 2))):

                        #get the bounding box around the pixel to send to the training
                        box = image[i-int(constants.IMG_SIZE / 2):i+int(constants.IMG_SIZE / 2),j-int(constants.IMG_SIZE / 2):j+int(constants.IMG_SIZE / 2)]

                        #append the box to a temporary array
                        tmp.append(box)

                        #once the temporary array is the same size as the batch size, run the testing on the batch
                        if(len(tmp) == constants.BATCH_SIZE or count == ((h - constants.IMG_SIZE) * (w - constants.IMG_SIZE)) - 1):
                            batch = np.array(tmp)
                            rawpredictions = out.eval({x1:batch, x2:batch})
                            mask = np.round(rawpredictions)

                            #now we go through the mask and insert the values to the correct position of best_guess which is a copy of
                            #the original image except all the values are -1
                            for raw,cat in zip(rawpredictions,mask):
                                best_guess[i0,j0] = cat
                                raw_guess[i0,j0] = raw
                                if j0 == (w - int(constants.IMG_SIZE/2)) - 1:
                                    j0 = int(constants.IMG_SIZE / 2)
                                    i0 += 1
                                else:
                                    j0 += 1

                            #give console output to show progress
                            outputResults(image[:,:,:3],np.array(best_guess),fout=seg_file)
                            np.save(os.path.splitext(logname)[0],raw_guess)
                            print('%i out of %i complete' % (count2,math.ceil(int((h - constants.IMG_SIZE) * (w - constants.IMG_SIZE) / constants.BATCH_SIZE))))
                            #empty tmporary array
                            tmp = []
                            count2 += 1
                        count += 1
        else:
            print("train ")
            print("trainseg ")
            print("test [image_filepath] [model_filepath]")
    else:
        print("oopsies")
        print("argv[1]: mode of operation (test,train)")

if __name__ == "__main__":
    tf.app.run()
