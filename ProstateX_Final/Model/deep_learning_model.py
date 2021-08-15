from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import  merge, UpSampling2D, Cropping2D
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.initializers import RandomNormal, VarianceScaling
import numpy as np
# Inspired by https://github.com/pietz/unet-keras (adds padding) and https://github.com/mirzaevinom/promise12_segmentation (adds variance scaling)
# For animations on convulotions : https://github.com/vdumoulin/conv_arithmetic is a great source


"""
	Useful definitions : 
	- Model groups layers into an object with training and inference features
	- Input() is used to instantiate a Keras tensor
	- concatenate() Functional interface to the Concatenate layer
	- Concatenate layer : Layer that concatenates a list of inputs
	- Conv2D : 2D convolutional layer
	- inc_rate: rate at which the conv channels will increase
	- MaxPooling2D : Global max pooling operation for spatial data
	- Conv2DTranspose : Transposed convolution layer (sometimes called Deconvolution)
	- USampling2D : Repeats the rows and columns of the data by size[0] and size[1] respectively
	- Cropping2D : Cropping layer for 2D input
	- RandomNormal : initializer that generates tensors with a normal distribution 
	- VarianceScaling : Initializer capable of adapting its scale to the shape of weights tensors
	- residual: add residual connections around each conv block if true

"""



"""

	[(W−K+2P)/S]+1.
	W is the input volume - in your case 128
	K is the Kernel size - in your case 5
	P is the padding - in your case 0 i believe
	S is the stride - which you have not provided.

"""

# Padding is not added in the orginial U-net. Its a method that helps avoiding the loss of information on corners of the image
# Here it was implemented in the source code I obtained, and I kept it for simplicity in coding (how I got it and it works), simplicity in dimensionality
# reduction and for the fact that it helps observe better spacial information. 
# https://stats.stackexchange.com/questions/246512/convolutional-layers-to-pad-or-not-to-pad for more details on why use padding

# variance scaling was used for kernel initialization (not used in the first source, but in the second)
# https://stats.stackexchange.com/questions/319323/whats-the-difference-between-variance-scaling-initializer-and-xavier-initialize/319849
# via the source above, variance scaling is better than the usual Xavier Initialization because we are using ReLu as an activation function

# why use an initialisation method? 
# via the papers that were written to develop these methods, they help 
# "The aim of weight initialization is to prevent layer activation outputs from exploding or vanishing 
# during the course of a forward pass through a deep neural network"
# https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79
# Variance scaling is used in the code source that I got from the github repo mentioned above. I am keeping it for the following reasons : 
# simplicity in coding, read online that it was useful, its working so why remove it?, I won't have time to test with and without, but it works so I let it here

# explanation on the residual networks part : https://stats.stackexchange.com/questions/321054/what-are-residual-connections-in-rnns


def conv_block(m, dim, acti, bn, res, do=0):

    init = VarianceScaling(scale=1.0/9.0) 
    n = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=init)(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, 3, activation=acti, padding='same', kernel_initializer=init)(n)
    n = BatchNormalization()(n) if bn else n

    return concatenate([n, m], axis=3) if res else n

def level_block(m, dim, depth, inc, acti, do, bn, mp, up, res):
	if depth > 0:
		n = conv_block(m, dim, acti, bn, res)
		m = MaxPooling2D()(n) if mp else Conv2D(dim, 3, strides=2, padding='same')(n)
		m = level_block(m, int(inc*dim), depth-1, inc, acti, do, bn, mp, up, res)
		if up:
			m = UpSampling2D()(m)
			m = Conv2D(dim, 2, activation=acti, padding='same')(m)
		else:
			m = Conv2DTranspose(dim, 3, strides=2, activation=acti, padding='same')(m)
		n = concatenate([n, m], axis=3)
		m = conv_block(n, dim, acti, bn, res)
	else:
		m = conv_block(m, dim, acti, bn, res, do)
	return m


def UNet(img_shape, out_ch=1, start_ch=64, depth=4, inc_rate=2., activation='relu',
		 dropout=0.0, batchnorm=False, maxpool=True, upconv=False, residual=False):

	i = Input(shape=img_shape)
	o = level_block(i, start_ch, depth, inc_rate, activation, dropout, batchnorm, maxpool, upconv, residual)
	o = Conv2D(out_ch, 1, activation='softmax')(o)
	return Model(inputs=i, outputs=o)


# model = UNet((128, 128, 7), start_ch=8, depth=4, out_ch=2, batchnorm=True, dropout=0.5, maxpool=True, residual=True)
# with depth 4, I get to an 8x8 image, with 830 000 trainable parameters
# with a depth of 5 i get a 4x4 image, which I think is a bit small. But of course, nothing is better than testing
# both models to see if it works better or not
# model.summary()

# I'm gonna try working with these paremters cause they give a reasonable amount of trainable paremterers, the depth seems
# good for the size of the image (goes down to 8x8) and the number of feature maps seems good. If it works well good, if not 
# we need to change them to obtain a good result. If they work, I won't have time to test out all the different paremters
# but I will stick with those since they work
# Its important : I chose all my parameters in a logical way based on the original paper
# , but other values of these parameters can be logical too.
# My goal isnt' to test out each parameter to get the best model, but to build a working two based models. That's 
# why I don't test each and every value
# For the depth of the network, in the original paper, the size of the image is 572x572 and gets down to 28x28 so a factor
# of 20 approx. Here, the image size is 128. Doing a factor 20 on 128 gives 6,4. So I had to choose between a depth of 5 which gives
# a 4x4 image and a depth of 4 which gives an 8x8 image. I think 4x4 is a bit small, but of course nothing is better than testing
# both and seeing what works better. In this master thesis, I won't have time to test every single value (since like I explained previously
# is is not the goal of my MT to get the best model). So I will go with a depth of 4, and if it doesn't work, I'll move up to a depth of 5
