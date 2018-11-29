
# coding: utf-8

# # Evading SVMs
# 
# In this section we will be training an SVM to distinguish between 0,1. The data is coming from the MNIST dataset which contains handwritten digits. We will be using the `scikit-learn` for the SVM training. 

# First let's start out by importing a few essentials

# In[ ]:


from sklearn import svm
import numpy as np
import os
import matplotlib.pyplot as plt


# Since we are only interessted in the 1s and 0s in the data we will need to pick those out. Since we need to this for botht the test and training data let's write a function for it.
# 
# This function does a few other things aswell. 
# - It normalizes the data, bringing it into the interval [0,1]
# - It is also only using a part of the data to makes things a bit faster
# - It also reshapes the data so we can use it with SVMs

# In[ ]:


def exract_ones_and_zeroes( data, labels ):
    data_zeroes = data[ np.argwhere( labels == 0 ) ][ :200 ]
    data_ones = data[ np.argwhere( labels == 1 ) ][ :200 ]
    x = np.vstack( (data_zeroes, data_ones) )

    x = x.reshape( (x.shape[ 0 ], -1) ) / 255.

    print( x.shape )

    labels_zeroes = np.zeros( data_zeroes.shape[ 0 ] ) - 1.0
    labels_ones = np.ones( data_ones.shape[ 0 ] )
    y = np.append( labels_zeroes, labels_ones )

    return x, y


# Next we need to load the data and spilt it into the correct parts

# In[ ]:


mnist_file = os.path.join( 'data', 'mnist', 'mnist.npz' )

# load the data
f = np.load( mnist_file )
x_train, y_train = f[ 'x_train' ], f[ 'y_train' ]
print( 'x_train', x_train.shape )
print( 'y_train', y_train.shape )

x_test, y_test = f[ 'x_test' ], f[ 'y_test' ]
print( 'x_test', x_test.shape )
print( 'y_test', y_test.shape )

f.close( )

# extract ones and zeroes
x_train, y_train = exract_ones_and_zeroes( x_train, y_train )
x_test, y_test = exract_ones_and_zeroes( x_test, y_test )


# We are going to define a SVM with a RFB kernel and train it. 
# Once training is done we are going to print the accuracy and show one of the images

# In[ ]:


clf = svm.SVC( )
clf.fit( x_train, y_train )
print( 'accuracy on test set:', clf.score( x_test, y_test ) )

# plot the first instance in the traning set
plt.imshow( x_test[ 0 ].reshape( 28, 28 ), cmap="gray_r" )
plt.axis( 'off' )
plt.show( )


# To evade the classifier we first pick a sample that we want to change. After that we need to retrive some of the parameters of the SVM which we will need to calculate the gradients

# In[ ]:


# constructing adversarial examples
sample = x_test[ 300 ]
print( 'class prediction for the test samples:', clf.predict( [ sample ] ) )

# Retrieve the internal parameters from the SVM
alpha = clf.dual_coef_
svs = clf.support_vectors_
nsv = svs.shape[ 0 ]
b = clf.intercept_

plt.imshow( sample.reshape( 28, 28 ), cmap="gray_r" )
plt.axis( 'off' )
plt.show( )


# Now that we have the internal parameters we can calcuate the gradients of the SVM and apply the modifications to t our selected sample

# In[ ]:


# the sample we modify
mod_sample = sample[ : ]

# lambda, strength of the modification
lmbd = 10.6

# Compute the kernel row matrix and kernel gradients for xc

kgrad = np.empty( svs.shape )
# do multiple update rounds
for n in range(1):
    # for all support vectors
    for i in range( nsv ):
        sv = svs[ i, : ]  # support vector x_i
        k = -2 * clf._gamma * np.exp( -clf._gamma * np.sqrt( np.sum( np.square( mod_sample - svs ) ) ) ) * (
                mod_sample - sv)
        dk = clf._gamma * k * (svs[ i, : ] - mod_sample)
        kgrad[ i, : ] = k

    grad = -1. * np.dot( alpha, kgrad )

    # modify the sample
    mod_sample = np.clip( mod_sample + lmbd * grad, 0.,1.)

mod_sample = np.clip( mod_sample, 0., 1. )
print( 'class prediction for the original sample:', clf.predict( [sample] ) )
print( 'class prediction for the modified sample:', clf.predict( mod_sample ) )
print( 'original sample:')
plt.imshow( sample.reshape( 28, 28 ), cmap="gray_r" )
plt.show( )
print( 'modified sample:')
plt.imshow( mod_sample.reshape( 28, 28 ), cmap="gray_r" )
plt.show( )


print( 'difference between the tow samples:')
plt.imshow( np.abs(sample-mod_sample).reshape( 28, 28 ), cmap="gray_r" )
plt.show( )

