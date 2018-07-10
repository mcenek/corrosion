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