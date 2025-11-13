import numpy as np

a = np.array([[[1, np.nan, 3],
              [np.nan, 2, 5]],[[1, np.nan, 3],
              [np.nan, 2, np.nan]]])

print(a.shape)
print(a[0,1,2])