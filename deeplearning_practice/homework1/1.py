from itertools import accumulate
import numpy as np
# a = [1,2,3]
# print(list(accumulate(a, initial = 0)))

def polar(x):
    if x.shape != (3,2):
        raise ValueError("x must be a 3x2 array")
    norm = np.linalg.norm(x, axis = 1).reshape(-1,1)
    print(norm.shape)
    theta = np.arctan2(x[:,1], x[:,0]).reshape(-1,1)
    ans = np.hstack((norm,theta))
    print("Cartesian:\n", x)
    print("Polar (r, theta):\n", ans)

m = np.asarray([[1,1],[1,-1],[1,np.sqrt(3)]])

polar(m)

