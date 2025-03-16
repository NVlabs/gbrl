import numpy as np

arr = np.zeros((100, 100), dtype=np.float32)
buffer = memoryview(arr)

print(buffer.format, buffer.shape)

