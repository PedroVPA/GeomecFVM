import numpy as np

I = np.zeros(4)

I[::2] = 2

for id in range(I.shape[0]):
    
    if id % 2 == 0:

        I[id] = 2








