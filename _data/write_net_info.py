
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
get_ipython().magic(u'matplotlib inline')

caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import caffe


# In[269]:

model_def = caffe_root + '/examples/mnist/lenet_batch1.prototxt'
model_weights = caffe_root + '/examples/mnist/lenet_iter_10000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


# In[270]:

import sys
import numpy as np
import lmdb
import caffe
import argparse
from matplotlib import pyplot
 
lmdbpath = '/home/liguangli/work/caffe/examples/mnist/mnist_test_lmdb'
env = lmdb.open(lmdbpath, readonly=True)
t = 0
with env.begin() as txn:
  cursor = txn.cursor()
  for key, value in cursor:
    print 'key: ',key
    datum = caffe.proto.caffe_pb2.Datum() #datum类型
    datum.ParseFromString(value) #转成datum
    flat_x = np.fromstring(datum.data, dtype=np.uint8) #转成numpy类型
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    y = datum.label#图片的label
    print flat_x.shape
    print type(flat_x)
    x2 = flat_x.reshape((28,28))
    print y
    
    fig = pyplot.figure()#把两张图片显示出来    
    pyplot.imshow(x2, cmap = plt.cm.gray)
    t = t + 1
    if t > 1 :
        break


# In[31]:

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = x

### perform classification
caffe.set_device(0)
caffe.set_mode_gpu()
output = net.forward()

output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()


# In[271]:

# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    print layer_name + '\t' + str(blob.data.shape)


# In[272]:

# (output_channels, input_channels, filter_height, filter_width)
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape)


# In[281]:

for layer in net.layer_dict:
    print layer
#for name in net.bottom_names:
    #print name


# In[55]:

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import caffe

caffe_root = '../'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

### load net model
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

### mean image
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

### input image
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

### set gpu
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()
net.forward() 


### output shape of layer >>> output.txt
# for each layer, show the output shape
for layer_name, blob in net.blobs.iteritems():
    f1 = open('/home/liguangli/split_net/caffenet/layer_output/' + layer_name + '.txt', 'w')
    print layer_name + '\t' + str(blob.data.shape)
    f1.write(layer_name + ' ' + str(blob.data.shape) + '\n')
    for element in blob.data.flat:
    #for element in np.nonzero(blob.data):
        f1.write(str(element) + " ")
    f1.write('\n')
    f1.close()


# In[56]:

### param shape of layer >>> param_shape.txt 
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape)


# In[15]:

import sys
import numpy as np
import lmdb
import caffe
import argparse
from matplotlib import pyplot
 
lmdbpath = '/home/liguangli/work/caffe/examples/mnist/mnist_test_lmdb'
env = lmdb.open(lmdbpath, readonly=True)
with env.begin() as txn:
  cursor = txn.cursor()
  for key, value in cursor:
    print 'key: ',key
    datum = caffe.proto.caffe_pb2.Datum() #datum类型
    datum.ParseFromString(value) #转成datum
    flat_x = np.fromstring(datum.data, dtype=np.uint8) #转成numpy类型
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    y = datum.label#图片的label
    print flat_x.shape
    print type(flat_x)
    x2 = flat_x.reshape((28,28))
    print y
    
    fig = pyplot.figure()#把两张图片显示出来    
    pyplot.imshow(x2, cmap = plt.cm.gray)
    break


# In[58]:

####### mnist-lenet

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import lmdb
import caffe
import argparse
get_ipython().magic(u'matplotlib inline')

caffe_root = '/home/liguangli/work/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

model_def = caffe_root + 'examples/mnist/lenet_batch1.prototxt'
model_weights = caffe_root + 'examples/mnist/lenet_iter_10000.caffemodel'
outputdir = '/home/liguangli/split_net/lenet'

# parse net
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# set gpu
caffe.set_device(0)
caffe.set_mode_gpu()

# write params
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape)
    
    f1 = open( outputdir + '/layer_param/' + layer_name + '.txt', 'w+') 
    f1.write(layer_name + " ")
    for temp in param[0].data.shape:
        f1.write(str(temp) + " ")
    f1.write("\n")
    for element in param[0].data.flat:
        f1.write("{0:.8f} ".format(element))
    f1.write('\n')
    f1.close()

# 1000 images forward

lmdbpath = '/home/liguangli/work/caffe/examples/mnist/mnist_test_lmdb'
env = lmdb.open(lmdbpath, readonly=True)
n = 1

# create output files
for layer_name, blob in net.blobs.iteritems():
    f2 = open(outputdir + '/layer_output/' + layer_name + '.txt', 'w+')
    f2.write(layer_name + " ")
    for temp in blob.data.shape:
        f2.write(str(temp) + " ")
    f2.write("\n")  
    f2.close()

with env.begin() as txn:
  cursor = txn.cursor()
  for key, value in cursor:
    #print 'key: ',key
    datum = caffe.proto.caffe_pb2.Datum() #datum类型
    datum.ParseFromString(value) #转成datum
    flat_x = np.fromstring(datum.data, dtype=np.uint8) #转成numpy类型
    x = flat_x.reshape(datum.channels, datum.height, datum.width)
    y = datum.label #图片的label

    net.blobs['data'].data[...] = x
    output = net.forward()
    
    for layer_name, blob in net.blobs.iteritems():
        
        f2 = open(outputdir + '/layer_output/' + layer_name + '.txt', 'a+')

        for element in blob.data.flat:
            f2.write("{0:.8f} ".format(element))
        f2.write('\n')
        f2.close()
    if n % 100 == 0:
        print "{0}/1000".format(n)
    n = n + 1
    if n > 1000 :
        break


# In[255]:

###imagenet-alexnet
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import lmdb
import caffe
import argparse
import cv2
get_ipython().magic(u'matplotlib inline')

caffe_root = '/home/liguangli/work/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

model_def = caffe_root + '/models/bvlc_alexnet/deploy_batch1.prototxt'
model_weights = caffe_root + '/models/bvlc_alexnet/bvlc_alexnet.caffemodel'
outputdir = '/home/liguangli/split_net/alexnet'

# parse net
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# set gpu
caffe.set_device(0)
caffe.set_mode_gpu()

# write params
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape)
    f1 = open( outputdir + '/layer_param/' + layer_name + '.txt', 'w+') 
    f1.write(layer_name + " ")
    for temp in param[0].data.shape:
        f1.write(str(temp) + " ")
    f1.write("\n")
    for element in param[0].data.flat:
        f1.write("{0:.8f} ".format(element))
    f1.write('\n')
    f1.close()

print "params done"
    
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# 1000 images forward

lmdbpath = '/dataset/ilsvrc12_val_lmdb'
env = lmdb.open(lmdbpath, readonly=True)
n = 1

# create output files
for layer_name, blob in net.blobs.iteritems():
    f2 = open(outputdir + '/layer_output/' + layer_name + '.txt', 'w+')
    f2.write(layer_name + " ")
    for temp in blob.data.shape:
        f2.write(str(temp) + " ")
    f2.write("\n")  
    f2.close()
    
print "output create done"
    
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        datum = caffe.proto.caffe_pb2.Datum() #datum类型
        datum.ParseFromString(value) #转成datum
        flat_x = np.fromstring(datum.data, dtype=np.uint8) #转成numpy类型
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        xx = np.transpose(x, (1,2,0))
        y = datum.label #图片的label
        xx = cv2.cvtColor(xx, cv2.COLOR_BGR2RGB) 
        transformed_image = transformer.preprocess('data', xx)
        
        #image = caffe.io.load_image(caffe_root + '/examples/images/cat.jpg')
        #transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        #print "done"

        if n % 10 == 0:
            print "{0}/100".format(n)
        for layer_name, blob in net.blobs.iteritems():

            f2 = open(outputdir + '/layer_output/' + layer_name + '.txt', 'a+')

            for element in blob.data.flat:
                f2.write("{0:.8f} ".format(element))
            f2.write('\n')
            f2.close()
        n = n + 1
        if n > 100:
            break


# In[259]:

###imagenet-googlenet
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import lmdb
import caffe
import argparse
import cv2
get_ipython().magic(u'matplotlib inline')

caffe_root = '/home/liguangli/work/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

model_def = caffe_root + '/models/bvlc_googlenet/deploy_batch1.prototxt'
model_weights = caffe_root + '/models/bvlc_googlenet/bvlc_googlenet.caffemodel'
outputdir = '/home/liguangli/split_net/googlenet'

# parse net
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# set gpu
caffe.set_device(0)
caffe.set_mode_gpu()

# write params
for layer_name, param in net.params.iteritems():
    print layer_name.replace('/', '_') + '\t' + str(param[0].data.shape)
    f1 = open( outputdir + '/layer_param/' + layer_name.replace('/', '_') + '.txt', 'w+') 
    f1.write(layer_name + " ")
    for temp in param[0].data.shape:
        f1.write(str(temp) + " ")
    f1.write("\n")
    for element in param[0].data.flat:
        f1.write("{0:.8f} ".format(element))
    f1.write('\n')
    f1.close()

print "params done"
    
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# 1000 images forward

lmdbpath = '/dataset/ilsvrc12_val_lmdb'
env = lmdb.open(lmdbpath, readonly=True)
n = 1

# create output files
for layer_name, blob in net.blobs.iteritems():
    f2 = open(outputdir + '/layer_output/' + layer_name.replace('/', '_') + '.txt', 'w+')
    f2.write(layer_name + " ")
    for temp in blob.data.shape:
        f2.write(str(temp) + " ")
    f2.write("\n")  
    f2.close()
    
print "output create done"
    
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        datum = caffe.proto.caffe_pb2.Datum() #datum类型
        datum.ParseFromString(value) #转成datum
        flat_x = np.fromstring(datum.data, dtype=np.uint8) #转成numpy类型
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        xx = np.transpose(x, (1,2,0))
        y = datum.label #图片的label
        xx = cv2.cvtColor(xx, cv2.COLOR_BGR2RGB) 
        transformed_image = transformer.preprocess('data', xx)
        
        #image = caffe.io.load_image(caffe_root + '/examples/images/cat.jpg')
        #transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        #print "done"

        if n % 10 == 0:
            print "{0}/100".format(n)
        for layer_name, blob in net.blobs.iteritems():

            f2 = open(outputdir + '/layer_output/' + layer_name.replace('/', '_') + '.txt', 'a+')

            for element in blob.data.flat:
                f2.write("{0:.8f} ".format(element))
            f2.write('\n')
            f2.close()
        n = n + 1
        if n > 100:
            break


# In[261]:

###imagenet-vgg16
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import lmdb
import caffe
import argparse
import cv2
get_ipython().magic(u'matplotlib inline')

caffe_root = '/home/liguangli/work/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

model_def = caffe_root + '/models/vgg16/VGG_ILSVRC_16_layers_deploy.prototxt'
model_weights = caffe_root + '/models/vgg16/VGG_ILSVRC_16_layers.caffemodel'
outputdir = '/home/liguangli/split_net/vgg16'

# parse net
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# set gpu
caffe.set_device(0)
caffe.set_mode_gpu()

# write params
for layer_name, param in net.params.iteritems():
    print layer_name.replace('/', '_') + '\t' + str(param[0].data.shape)
    f1 = open( outputdir + '/layer_param/' + layer_name.replace('/', '_') + '.txt', 'w+') 
    f1.write(layer_name + " ")
    for temp in param[0].data.shape:
        f1.write(str(temp) + " ")
    f1.write("\n")
    for element in param[0].data.flat:
        f1.write("{0:.8f} ".format(element))
    f1.write('\n')
    f1.close()

print "params done"
    
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# 1000 images forward

lmdbpath = '/dataset/ilsvrc12_val_lmdb'
env = lmdb.open(lmdbpath, readonly=True)
n = 1

# create output files
for layer_name, blob in net.blobs.iteritems():
    f2 = open(outputdir + '/layer_output/' + layer_name.replace('/', '_') + '.txt', 'w+')
    f2.write(layer_name + " ")
    for temp in blob.data.shape:
        f2.write(str(temp) + " ")
    f2.write("\n")  
    f2.close()
    
print "output create done"
    
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        datum = caffe.proto.caffe_pb2.Datum() #datum类型
        datum.ParseFromString(value) #转成datum
        flat_x = np.fromstring(datum.data, dtype=np.uint8) #转成numpy类型
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        xx = np.transpose(x, (1,2,0))
        y = datum.label #图片的label
        xx = cv2.cvtColor(xx, cv2.COLOR_BGR2RGB) 
        transformed_image = transformer.preprocess('data', xx)
        
        #image = caffe.io.load_image(caffe_root + '/examples/images/cat.jpg')
        #transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        #print "done"

        if n % 10 == 0:
            print "{0}/100".format(n)
        for layer_name, blob in net.blobs.iteritems():

            f2 = open(outputdir + '/layer_output/' + layer_name.replace('/', '_') + '.txt', 'a+')

            for element in blob.data.flat:
                f2.write("{0:.8f} ".format(element))
            f2.write('\n')
            f2.close()
        n = n + 1
        if n > 100:
            break


# In[262]:

###imagenet-resnet-18
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import lmdb
import caffe
import argparse
import cv2
get_ipython().magic(u'matplotlib inline')

caffe_root = '/home/liguangli/work/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

model_def = caffe_root + '/models/resnet-18/test_batch1.prototxt'
model_weights = caffe_root + '/models/resnet-18/resnet-18.caffemodel'
outputdir = '/home/liguangli/split_net/resnet-18'

# parse net
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# set gpu
caffe.set_device(0)
caffe.set_mode_gpu()

# write params
for layer_name, param in net.params.iteritems():
    print layer_name.replace('/', '_') + '\t' + str(param[0].data.shape)
    f1 = open( outputdir + '/layer_param/' + layer_name.replace('/', '_') + '.txt', 'w+') 
    f1.write(layer_name + " ")
    for temp in param[0].data.shape:
        f1.write(str(temp) + " ")
    f1.write("\n")
    for element in param[0].data.flat:
        f1.write("{0:.8f} ".format(element))
    f1.write('\n')
    f1.close()

print "params done"
    
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# 1000 images forward

lmdbpath = '/dataset/ilsvrc12_val_lmdb'
env = lmdb.open(lmdbpath, readonly=True)
n = 1

# create output files
for layer_name, blob in net.blobs.iteritems():
    f2 = open(outputdir + '/layer_output/' + layer_name.replace('/', '_') + '.txt', 'w+')
    f2.write(layer_name + " ")
    for temp in blob.data.shape:
        f2.write(str(temp) + " ")
    f2.write("\n")  
    f2.close()
    
print "output create done"
    
with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        datum = caffe.proto.caffe_pb2.Datum() #datum类型
        datum.ParseFromString(value) #转成datum
        flat_x = np.fromstring(datum.data, dtype=np.uint8) #转成numpy类型
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        xx = np.transpose(x, (1,2,0))
        y = datum.label #图片的label
        xx = cv2.cvtColor(xx, cv2.COLOR_BGR2RGB) 
        transformed_image = transformer.preprocess('data', xx)
        
        #image = caffe.io.load_image(caffe_root + '/examples/images/cat.jpg')
        #transformed_image = transformer.preprocess('data', image)
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()
        #print "done"

        if n % 10 == 0:
            print "{0}/100".format(n)
        for layer_name, blob in net.blobs.iteritems():

            f2 = open(outputdir + '/layer_output/' + layer_name.replace('/', '_') + '.txt', 'a+')

            for element in blob.data.flat:
                f2.write("{0:.8f} ".format(element))
            f2.write('\n')
            f2.close()
        n = n + 1
        if n > 100:
            break


# In[233]:

###imagenet-test
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import lmdb
import caffe
import argparse
import cv2
get_ipython().magic(u'matplotlib inline')

caffe_root = '/home/liguangli/work/caffe'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')

model_def = caffe_root + '/models/bvlc_alexnet/deploy_batch1.prototxt'
model_weights = caffe_root + '/models/bvlc_alexnet/bvlc_alexnet.caffemodel'

# parse net
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# set gpu
caffe.set_device(0)
caffe.set_mode_gpu()

# write params
for layer_name, param in net.params.iteritems():
    print layer_name + '\t' + str(param[0].data.shape)

print "params done"
    
# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# 1000 images forward

lmdbpath = '/dataset/ilsvrc12_val_lmdb'
env = lmdb.open(lmdbpath, readonly=True)
n = 1

with env.begin() as txn:
    cursor = txn.cursor()
    for key, value in cursor:
        datum = caffe.proto.caffe_pb2.Datum() #datum类型
        datum.ParseFromString(value) #转成datum
        flat_x = np.fromstring(datum.data, dtype=np.uint8) #转成numpy类型
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        xx = np.transpose(x, (1,2,0))
        y = datum.label #图片的label
        print y
        xx = cv2.cvtColor(xx, cv2.COLOR_BGR2RGB) 
        pyplot.imshow(xx)
        transformed_image = transformer.preprocess('data', xx)
        
        net.blobs['data'].data[...] = transformed_image

        output = net.forward()

        output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

        print 'predicted class is:', output_prob.argmax()

        # load ImageNet labels
        labels_file = caffe_root + '/data/ilsvrc12/synset_words.txt'
        if not os.path.exists(labels_file):
            get_ipython().system(u'../data/ilsvrc12/get_ilsvrc_aux.sh')

        labels = np.loadtxt(labels_file, str, delimiter='\t')

        print 'output label:', labels[output_prob.argmax()]
        
        n = n + 1
        if n > 1:
            break


# In[263]:

n = 1
for layer_name, blob in net.blobs.iteritems():
    print str(n) + " " + layer_name + str(blob.data.shape)
    if n == 1:
        print max(blob.data.flat)
        print min(blob.data.flat)
        pyplot.hist(blob.data.flat,100)
    n = n + 1


# In[ ]:



