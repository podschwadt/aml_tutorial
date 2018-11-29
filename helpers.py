import numpy as np
import os
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import matplotlib.pyplot as plt


def exract_ones_and_zeroes( data, labels ):
    # data_zeroes = data[ np.argwhere( labels == 0 ) ]
    # data_ones = data[ np.argwhere( labels == 1 ) ]
    data_zeroes = data[ np.argwhere( labels == 0 ).reshape( -1 ) ][ :200 ]
    print( data_zeroes.shape )
    data_ones = data[ np.argwhere( labels == 1 ).reshape( -1 ) ][ :200 ]
    x = np.vstack( (data_zeroes, data_ones) )

    x = x / 255.

    print( x.shape )

    labels_zeroes = np.zeros( data_zeroes.shape[ 0 ] )
    labels_ones = np.ones( data_ones.shape[ 0 ] )
    y = np.append( labels_zeroes, labels_ones )

    return x, y


def load_mnist( ):
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
    return (x_train, y_train), (x_test, y_test)


def convert_to_keras_image_format( x_train, x_test ):
    if keras.backend.image_data_format( ) == 'channels_first':
        x_train = x_train.reshape( x_train.shape[ 0 ], 1, x_train.shape[ 1 ], x_train.shape[ 2 ] )
        x_test = x_test.reshape( x_test.shape[ 0 ], 1, x_train.shape[ 1 ], x_train.shape[ 2 ] )
    else:
        x_train = x_train.reshape( x_train.shape[ 0 ], x_train.shape[ 1 ], x_train.shape[ 2 ], 1 )
        x_test = x_test.reshape( x_test.shape[ 0 ], x_train.shape[ 1 ], x_train.shape[ 2 ], 1 )

    return x_train, x_test


def mnist_cnn_model( x_train, y_train, x_test, y_test, epochs=2 ):
    # define the classifier
    clf = keras.Sequential( )
    clf.add( Conv2D( 32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[ 1: ] ) )
    clf.add( Conv2D( 64, (3, 3), activation='relu' ) )
    clf.add( MaxPooling2D( pool_size=(2, 2) ) )
    clf.add( Dropout( 0.25 ) )
    clf.add( Flatten( ) )
    clf.add( Dense( 128, activation='relu' ) )
    clf.add( Dropout( 0.5 ) )
    clf.add( Dense( y_train.shape[ 1 ], activation='softmax' ) )

    clf.compile( loss=keras.losses.categorical_crossentropy,
                 optimizer='adam',
                 metrics=[ 'accuracy' ] )

    clf.fit( x_train, y_train,
             epochs=epochs,
             verbose=1 )
    clf.summary( )
    score = clf.evaluate( x_test, y_test )
    print( 'Test loss:', score[ 0 ] )
    print( 'Test accuracy:', score[ 1 ] )

    return clf


def show_image( img ):
    plt.imshow( img.reshape( 28, 28 ), cmap="gray_r" )
    plt.axis( 'off' )
    plt.show( )
