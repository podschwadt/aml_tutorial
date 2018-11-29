import numpy as np
import os
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import BasicIterativeMethod, FastGradientMethod, CarliniWagnerL2
import matplotlib.pyplot as plt
import tensorflow as tf


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

# we need to bring the data in to a format that our cnn likes
y_train = keras.utils.to_categorical( y_train, 2 )
y_test = keras.utils.to_categorical( y_test, 2 )

if keras.backend.image_data_format( ) == 'channels_first':
    x_train = x_train.reshape( x_train.shape[ 0 ], 1, x_train.shape[ 1 ], x_train.shape[ 2 ] )
    x_test = x_test.reshape( x_test.shape[ 0 ], 1, x_train.shape[ 1 ], x_train.shape[ 2 ] )
    input_shape = (1, x_train.shape[ 1 ], x_train.shape[ 2 ])
else:
    x_train = x_train.reshape( x_train.shape[ 0 ], x_train.shape[ 1 ], x_train.shape[ 2 ], 1 )
    x_test = x_test.reshape( x_test.shape[ 0 ], x_train.shape[ 1 ], x_train.shape[ 2 ], 1 )
    input_shape = (x_train.shape[ 1 ], x_train.shape[ 2 ], 1)

# need to some setup so everything gets excturted in the same tensorflow session

session = tf.Session( )
keras.backend.set_session( session )

# define the classifier
clf = keras.Sequential( )
clf.add( Conv2D( 32, kernel_size=(3, 3), activation='relu', input_shape=input_shape ) )
clf.add( Conv2D( 64, (3, 3), activation='relu' ) )
clf.add( MaxPooling2D( pool_size=(2, 2) ) )
clf.add( Dropout( 0.25 ) )
clf.add( Flatten( ) )
clf.add( Dense( 128, activation='relu' ) )
clf.add( Dropout( 0.5 ) )
clf.add( Dense( 2, activation='softmax' ) )

clf.compile( loss=keras.losses.categorical_crossentropy,
             optimizer='adam',
             metrics=[ 'accuracy' ] )

clf.fit( x_train, y_train,
         epochs=2,
         verbose=1 )
clf.summary( )
score = clf.evaluate( x_test, y_test )
print( 'Test loss:', score[ 0 ] )
print( 'Test accuracy:', score[ 1 ] )

# plot the first instance in the traning set
plt.imshow( x_test[ 0 ].reshape( 28, 28 ), cmap="gray_r" )
plt.axis( 'off' )
plt.show( )

# constructing adversarial examples
# picking a test sample
sample = x_test[ 1, : ]
print( 'class prediction for the test samples:',
       clf.predict( sample.reshape( (1, sample.shape[ 0 ], sample.shape[ 1 ], sample.shape[ 2 ]) ) ) )
# setup the attack
wrapper = KerasModelWrapper( clf )
fgm = FastGradientMethod( wrapper, sess=session )
eps = 0.3  # allowed maximum modification

# excetute the attack
modified_sample = fgm.generate_np( sample.reshape( (1, sample.shape[ 0 ], sample.shape[ 1 ], sample.shape[ 2 ]) ),
                                   **{ 'eps': eps } )

print( 'class prediction for the modified test samples:',
       clf.predict( modified_sample.reshape( (1, sample.shape[ 0 ], sample.shape[ 1 ], sample.shape[ 2 ]) ) ) )
plt.imshow( modified_sample.reshape( 28, 28 ), cmap="gray_r" )
plt.axis( 'off' )
plt.show( )

# let's try a stronger attack
cw_l2 = CarliniWagnerL2( wrapper, sess=session )
modified_sample = cw_l2.generate_np( sample.reshape( (1, sample.shape[ 0 ], sample.shape[ 1 ], sample.shape[ 2 ]) ),
                                     **{ 'eps': eps } )

print( 'class prediction for the cw modified test samples:',
       clf.predict( modified_sample.reshape( (1, sample.shape[ 0 ], sample.shape[ 1 ], sample.shape[ 2 ]) ) ) )
plt.imshow( modified_sample.reshape( 28, 28 ), cmap="gray_r" )
plt.axis( 'off' )
plt.show( )
