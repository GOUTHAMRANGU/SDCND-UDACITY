import os
import numpy as np
import pickle
from  scipy import ndimage
x=os.getcwd()
y=os.listdir(x+'/down')
print(y)
dataset = np.ndarray(shape=(5,32,32,3))
for i in range(len(y)):
	image=ndimage.imread(x+'/down/'+y[i]).astype(float)
	print(image.shape)
	dataset[i,:,:,:]=image
pickle.dump( dataset, open( "dataset.p", "wb" ) )


