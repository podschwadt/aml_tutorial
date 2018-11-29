from sklearn import svm
import gzip
import numpy as np
import struct
import os
import matplotlib.pyplot as plt


def exract_ones_and_zeroes( data, labels ):
    # data_zeroes = data[ np.argwhere( labels == 0 ) ]
    # data_ones = data[ np.argwhere( labels == 1 ) ]
    data_zeroes = data[ np.argwhere( labels == 0 ) ][ :200 ] - 1.0
    data_ones = data[ np.argwhere( labels == 1 ) ][ :200 ]
    x = np.vstack( (data_zeroes, data_ones) )

    x = x.reshape( (x.shape[ 0 ], -1) ) / 255.

    print( x.shape )

    labels_zeroes = np.zeros( data_zeroes.shape[ 0 ] ) - 1.0
    labels_ones = np.ones( data_ones.shape[ 0 ] )
    y = np.append( labels_zeroes, labels_ones )

    return x, y


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

clf = svm.SVC( )

clf.fit( x_train, y_train )

print( 'accuracy on test set:', clf.score( x_test, y_test ) )

# plot the first instance in the traning set
plt.imshow( x_test[ 0 ].reshape( 28, 28 ), cmap="gray_r" )
plt.axis( 'off' )
plt.show( )

# print( "Image Label: ", train[ sample_number, -1 ] )

# constructing adversarial examples
sample = x_test[ 1 ]
print( 'class prediction for the test samples:', clf.predict( [ sample ] ) )

# Retrieve the internal parameters from the SVM
alpha = clf.dual_coef_
svs = clf.support_vectors_
nsv = svs.shape[ 0 ]
b = clf.intercept_

# the sample we modify
mod_sample = sample[ : ]

# Compute the kernel row matrix and kernel gradients for xc
kgrad = np.empty( svs.shape )
# for all support vectors
for n in range( 1000 ):
    for i in range( nsv ):
        sv = svs[ i, : ]  # support vector x_i
        k = -2 * clf._gamma * np.exp( -clf._gamma * np.sqrt( np.sum( np.square( mod_sample - svs ) ) ) ) * (
                mod_sample - sv)
        dk = clf._gamma * k * (svs[ i, : ] - mod_sample)
        kgrad[ i, : ] = k

    grad = -1. * np.dot( alpha, kgrad )

    # modify the sample
    mod_sample = mod_sample - grad
    print( 'class prediction for the test samples after {} midfication epochs:'.format( n ), clf.predict( mod_sample ) )

mod_sample = np.clip( mod_sample, 0, 1 ) * 255
print( 'class prediction for the test samples:', clf.predict( mod_sample ) )
plt.imshow( mod_sample[ 0 ].reshape( 28, 28 ).astype( int ), cmap="gray_r" )
plt.show( )
