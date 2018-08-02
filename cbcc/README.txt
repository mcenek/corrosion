How to run each python script:

feature.py:

	python feature.py # note: this does not work at the moment, as I just have it running per run on the network

nn.py:

	python nn.py <training.txt> <test.txt> <image.jpg> "Network.h5"
	or 
	python nn.py <image.jpg> <network.h5>

	# training and test are file created by pixel_selection.py

mass_train.py
	
	python mass_train.py <groundtruths/> <images/> <network.h5>
	# ground truths and images are folders of the groundtruths and images for those groundtruths

visualize_texture.py

	python visualize_texture.py <image.jpg>

pixel_selection.py

	python pixel_selection.py <groundtruth.jpg>


split_nn.py:

	python split_nn.py <groundtruth.jpg> <image.jpg> "Network.h5" <sdmatrix.npy> # To train a new network
	or
	python split_nn.py <image.jpg> <network.h5>  <sdmatrix.npy> # To run a network on an image
	# Note: The network name provided must just be the network name, nothing before or after, as the there are actually three networks that will be created
	# by this program, all of which will be prefixed with, "color_" "texture_" or "combine_", so if the input is not exactly the same as when you created
	# the network, then it will be unable to load the networks properly

mass_train_split.py:

	python mass_train_split.py <groundtruths/> <images/> <network.h5> <sdmatrices/>
	# Same notes as with split_nn.py and mass_train.py

signed_differences.py:

	python signed_differences.py <images/>