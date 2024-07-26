#%pylab
#%matplotlib inline
from matplotlib import pyplot as plt
from numpy import *
import random
random.seed(12)

import mnist
mnist.datasets_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
# take only certain digits
Digits = array([0,1,2])
nDigits = len(Digits)

labels = mnist.train_labels()
digs = []
for i in range(len(labels)):
    if labels[i] in Digits:
        digs.append(i)

labels = labels[digs]
images = mnist.train_images()[digs,4:-4,4:-4]   # start with 20X20 pixel images

ex = 2

fig, axes = plt.subplots(1,2)
#fig, axes = subplots(1,2)
#axes[0].imshow(images[ex,:,:] * -1, cmap='gray')
axes[0].imshow(images[ex,:,:] * -1, cmap='gray')
axes[0].set_title('20X20 pixels')
print(labels[ex])

from skimage.measure import block_reduce
Dimages = block_reduce(images, (1,2,2), func=mean)  # downsample images to 10X10 pixels
axes[1].imshow(Dimages[ex,:,:] * -1, cmap='gray')
axes[1].set_title('10X10 pixels')

plt.show()
