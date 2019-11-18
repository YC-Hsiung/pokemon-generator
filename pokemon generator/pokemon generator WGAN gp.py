#!/usr/bin/env python
# coding: utf-8

# In[11]:

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
def loadpm():
    pmimg=[]
    idx=1
    while True:
        idx=str(idx)
        imgpath='pics/type/grass/40x40/poke'+idx+'.jpg'
        
        try:
            img=plt.imread(imgpath)
        except:
            break
        if(idx=='1'):
            pmimg=img.reshape(1,img.shape[0], img.shape[1], 3)
        else:
            pmimg =np.vstack((pmimg,img.reshape(1,img.shape[0], img.shape[1], 3)))
        idx=int(idx)
        idx+=1
    return pmimg


# In[12]:


pokemon=loadpm()
pokemon=(pokemon-127.5)/127.5
print((pokemon.shape))


# In[3]:


import os
import numpy as np
import tensorflow as tf
from IPython.core.debugger import Tracer
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,Conv2D,MaxPooling2D,Conv2DTranspose
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential,Model
from keras.optimizers import Adam,RMSprop
import keras.backend as K
from keras.engine.topology import Layer
import matplotlib.pyplot as plt
from keras.layers.merge import _Merge

class Subtract(_Merge):
    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output-inputs[i]
        return output
class GradNorm(Layer):
    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


# In[16]:


class Generator():
    def __init__(self,gen_dim,output_size):
        self.network=Sequential()
        element=1
        for i in output_size:
            element*=i
            
        IsDense=False
        if(IsDense):
        #Dense
            self.network.add(Dense(element//16,input_dim=gen_dim,activation='relu'))
            self.network.add(Dense(element//4,activation='relu'))
            self.network.add(Dense(element//4,activation='relu'))
            self.network.add(Dense(element//1,activation='tanh'))
            self.network.add(Reshape(output_size))
        else:
        #Convolution
            self.network.add(Dense(element//16*256,input_dim=gen_dim,activation='relu'))
            self.network.add(Reshape((output_size[0]//4,output_size[1]//4,output_size[2]*256)))
            self.network.add(Conv2DTranspose(filters=256,kernel_size=(3,3),strides=2,padding='same',activation='relu'))  
            self.network.add(Conv2DTranspose(filters=256,kernel_size=(3,3),strides=2,padding='same',activation='relu'))
            self.network.add(Conv2DTranspose(filters=3,kernel_size=(5,5),padding='same',activation='tanh'))
        self.network.summary()
class Discriminator():
    #Critic
    def __init__(self,input_size):
        self.network=Sequential()
        element=1
        for i in input_size:
            element*=i

        self.network.add(Conv2D(filters=256,kernel_size=(5,5),padding='same',input_shape=input_size))
        self.network.add(LeakyReLU(0.2))
        self.network.add(Conv2D(filters=256,kernel_size=(3,3),padding='same'))
        self.network.add(LeakyReLU(0.2))
        
        self.network.add((Flatten()))
        self.network.add(Dropout(0.2))
        #self.network.add(Dense(50))
        self.network.add(Dense(1,activation='linear'))
        
        self.network.summary()
        
        
        
class GAN():
    def __init__(self,trainingset,gen_dim):
        self.trainshape=trainingset.shape
        self.trainingset=trainingset
        self.gen_dim=gen_dim
        
        self.n_discriminator = 5
        
        self._lambda = 10
        optimizer = Adam(lr=0.0001,beta_1=0,beta_2=0.9)#RMSprop(lr=0.00005)
        #gen_dis
        self.generator=Generator(gen_dim,(trainingset.shape[1],trainingset.shape[2],trainingset.shape[3]))
        self.discriminator=Discriminator((trainingset.shape[1],trainingset.shape[2],trainingset.shape[3]))
        self.GAN_input=Input((gen_dim,))
        self.GAN_output=self.discriminator.network(self.generator.network(self.GAN_input))
        self.network=Model(self.GAN_input,self.GAN_output)
        self.network.compile(loss=self.wasserstein_loss,optimizer=optimizer)
        
        #gradient
        self.inputshape=self.trainshape[1:]
        self.real_input,self.gen_input,self.mixed_input=Input(self.inputshape),Input(self.inputshape),Input(self.inputshape)
        
        self.sub=Subtract()([self.discriminator.network(self.gen_input),self.discriminator.network(self.real_input)])
        self.grad_norm=GradNorm()([self.discriminator.network(self.mixed_input),self.mixed_input])
        self.grad=Model([self.gen_input,self.real_input,self.mixed_input],[self.sub,self.grad_norm])
        self.grad.compile(optimizer=optimizer,loss=[self.mean_loss,'MSE'],loss_weights=[1,self._lambda])
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    def mean_loss(self, y_true, y_pred):
        return K.mean(y_pred)
    def train(self,iterations):
        #train discriminator
        #self.start=0
        self.batch_size=20
        #self.more_to_stop=0
        self.real_label=np.ones((self.batch_size, 1))
        self.gen_label=-np.ones((self.batch_size, 1))
        for step in range(iterations):
        #train discriminator
            for i in range(self.n_discriminator):
            #prepare img
                self.random_gen_vectors=np.random.normal(size=(self.batch_size,self.gen_dim))
                self.gen_img=self.generator.network.predict(self.random_gen_vectors)
                self.epsilon=np.random.uniform(0,1,size=(self.batch_size,1,1,1))
                self.idx=np.random.randint(0,self.trainingset.shape[0],self.batch_size)
                self.real_img=self.trainingset[self.idx]
                self.mixed=self.epsilon*self.real_img+(1-self.epsilon)*self.gen_img
                
        
                #self.labels -= 0.1 * np.random.rand(self.labels.shape[0],self.labels.shape[1])
                self.discriminator.network.trainable = True
                self.d_loss = self.grad.train_on_batch([self.gen_img,self.real_img,self.mixed],[np.ones((self.batch_size,1)),np.ones((self.batch_size,1))])
                
                
        #train generator
            self.discriminator.network.trainable = False
            self.random_gen_vector=np.random.normal(size=(self.batch_size,self.gen_dim))
            self.g_loss=self.network.train_on_batch(self.random_gen_vector,self.gen_label)
            print('iter:%d,d_loss=%f,g_loss=%f'%(step,self.d_loss[0],self.g_loss))
            if step%50==0:
                #save image
                random_vector=np.random.normal(size=(25,self.gen_dim))
                images=self.generator.network.predict(random_vector)
                images=(images+1)*127.5
                plt.figure(figsize=(10, 10))
                for i in range(images.shape[0]):
                    plt.subplot(5,5,i+1)
                    image = images[i, :, :, :]
                    image = np.reshape(image,(self.trainshape[1:]))
                    plt.imshow(image.astype(np.uint8))
                    plt.axis('off')
                plt.tight_layout()
                plt.savefig("output/poke_%d.png" % step)
                plt.close('all')
                    


# In[ ]:


gan=GAN(pokemon,20)
gan.train(20000)


# In[6]:


'''
import numpy as np
from IPython.core.debugger import Tracer
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential,Model,load_model

import keras.backend as K
from keras.engine.topology import Layer
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
def wasserstein_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)
gan=load_model('eevee2_model.h5', custom_objects={'wasserstein_loss': wasserstein_loss})
#gan.summary()
generator=gan.layers[1]
'''
