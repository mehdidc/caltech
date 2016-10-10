from lib import linreg_tf, linreg_np
import time
from sklearn.datasets import make_regression
X, y = make_regression(1000, 1000)

t = time.time()
for _ in range(100):
    linreg_tf(X, y)
print(time.time() - t)

t = time.time()
for _ in range(100):
    linreg_np(X, y)
print(time.time() - t)
