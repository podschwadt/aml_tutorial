
# coding: utf-8

# # Defense with adversarial training
# 
# In this section we will use adversarial training to harden our CNN against adversarial examples. 
# 
# In adversarial training the dataset get "augmented" with adversarial examples that are correctly labeled. This way the network learns that such pertubations are possible and can adapt to them. 
# 
# We will be using the IBM Adversarial Robustness Toolbox in this exercise. It does offer a very easy to use implementation of adversarial training and bunch of other defenses. 
# https://github.com/IBM/adversarial-robustness-toolbox
# 
# 
# We start out by importing most of the things we need. The `helpers` module contains a bunch of code that we have used in earlier exersices. They functions have been moved there to keep it shorter

# In[ ]:


# most of our imports
import warnings
import numpy as np
import os
with warnings.catch_warnings():
    import keras # keras is still using some deprectade code
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import BasicIterativeMethod, FastGradientMethod, CarliniWagnerL2
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
from art.classifiers import KerasClassifier
from helpers import load_mnist, exract_ones_and_zeroes, convert_to_keras_image_format, mnist_cnn_model


# We start out by loading the data, preparing it and training our CNN

# In[ ]:


# load the data
(x_train, y_train), (x_test, y_test)= load_mnist()

# extract ones and zeroes
x_train, y_train = exract_ones_and_zeroes( x_train, y_train )
x_test, y_test = exract_ones_and_zeroes( x_test, y_test )

# we need to bring the data in to a format that our cnn likes
y_train = keras.utils.to_categorical( y_train, 2 )
y_test = keras.utils.to_categorical( y_test, 2 )

# convert it to a format keras can work with
x_train, x_test = convert_to_keras_image_format(x_train, x_test)

# need to some setup so everything gets excturted in the same tensorflow session
session = tf.Session( )
keras.backend.set_session( session )

# get and train our cnn
clf = mnist_cnn_model( x_train, y_train, x_test, y_test)


# We want to know how robust our model is against an attack. To do this we are calculating the `empirical robustness` This is equivalent to computing the minimal perturbation that the attacker must introduce for a    successful attack. Paper link: https://arxiv.org/abs/1511.04599

# In[ ]:


from art.metrics import empirical_robustness

# wrap the model an calculte emperical robustnees
wrapper = KerasClassifier(clip_values=(0,1), model=clf)
print( 'robustness of the undefended model', 
      empirical_robustness( wrapper, x_test, 'fgsm', {}))


# Let's create an adversarial example and see how it looks

# In[ ]:


# create an adversarial example with fgsm and plot it
from art.attacks import FastGradientMethod
fgsm = FastGradientMethod( wrapper )
x_adv = fgsm.generate(x_test[0].reshape((1,28,28,1) ))
print( 'class prediction for the adversarial sample:',
       clf.predict( x_adv.reshape((1,28,28,1) ) ) 
     )
plt.imshow( x_adv.reshape( 28, 28 ), cmap="gray_r" )
plt.axis( 'off' )
plt.show( )



# ## Adversarial Training
# 
# We are getting new untrained model with the same architecture that we have been using so far. 
# 
# To use the adversarial training that comes with `art` we need to pass our wrapped model to and `AdversarialTrainer` instance. The `AdversarialTrainer` also needs an instance of the attack that will be used to create the adversarial exmaples.

# In[ ]:


from art.defences import AdversarialTrainer

# get a new untrained model and warp it
new_model = mnist_cnn_model( x_train, y_train, x_test, y_test, epochs=0 )
defended_model = KerasClassifier(clip_values=(0,1), model=new_model )
# define the attack we are using
fgsm = FastGradientMethod( defended_model )


# Create the `AdversarialTrainer` instance. 
# Train the model and evaluate it on the test data.

# In[ ]:


# define the adversarial trainer and train the new network
adversarial_tranier = AdversarialTrainer( defended_model, fgsm )
adversarial_tranier.fit( x_train, y_train, batch_size=100, nb_epochs=2 )

# evaluate how good our model is
defended_model._model.evaluate( x_test,y_test )


# Calculate the `empirical robustness` for our now hopfully more robust model

# In[ ]:


# calculate the empiracal robustness
print( 'robustness of the defended model', 
      empirical_robustness( defended_model, x_test[0:], 'fgsm', {}) )
x_adv = fgsm.generate(x_test[0].reshape((1,28,28,1) ))
print( 'class prediction for the adversarial sample:',
       clf.predict( x_adv.reshape((1,28,28,1) ) ) 
     )
plt.imshow( x_adv.reshape( 28, 28 ), cmap="gray_r" )
plt.axis( 'off' )
plt.show( )

