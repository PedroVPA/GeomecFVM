from tracemalloc import start
import numpy as np
import time

val = np.arange(10000000)

start = time.time()

list1 = []
list1.extend(val)

print(f"t1 = {time.time() - start}")

start = time.time()

array = np.array([])
shape1 = array.shape[0]
shape2 = val.shape[0]
array_new = np.zeros(shape1 + shape2)
id1 = np.arange(shape1)
id2 = np.arange(shape2) + shape1
array_new[id1] = array
array_new[id2] = val
array = array_new.astype('int')

print(f"t2 = {time.time() - start}")

print('lmao')
