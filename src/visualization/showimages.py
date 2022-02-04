import torch
from data import mnist
from torchvision import transforms
from matplotlib import pyplot
import random

#
# Program that displays a random batch of images.
#

trainloader, _ = mnist()
dataiter = iter(trainloader)

batch = random.randint(0, 78)
print(f'Batch #{batch}')

for i in range(batch):
  images, labels = dataiter.next()

for i in range(64):  
    pyplot.subplot(8, 8, i+1)
    pyplot.axis('off')
    pyplot.imshow(images[i].numpy().squeeze(), cmap=pyplot.get_cmap('gray'))
pyplot.show()
