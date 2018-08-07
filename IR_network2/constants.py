

KEEP_RATE = 0.8                     #Rate of dropping out in the dropout layer
LOG_DIR = "./logs"             #Directory where the logs would be stored for visualization of the training

#Neural network constants
CAT_DIR = "../Corrosion_Dataset/IR_Images/"
LABEL_DIR = "../Corrosion_Dataset/IR_Images_Ground_Truths/"
TEST_CATDIR = "../Corrosion_Dataset/IRtest/defects/"
TEST_LABELDIR = "../Corrosion_Dataset/IRtest/labels/"
CAT1_ONEHOT     = [1,0]
CAT2_ONEHOT     = [0,1]


LEARNING_RATE = 0.001               #Learning rate for training the CNN
CNN_LOCAL1 = 32                  #Number of features output for conv layer 1
CNN_GLOBAL = 32                  #Number of features output for conv layer 1
CLASSES      = 2
CNN_EPOCHS       = 2000
CNN_FULL1   = 200                #Number of features output for fully connected layer1
FULL_IMGSIZE = 500
IMG_SIZE = 30
IMG_DEPTH   = 5
BATCH_SIZE = 2000

MIN_DENSITY = 10000
SPATIAL_RADIUS = 5
RANGE_RADIUS = 5


