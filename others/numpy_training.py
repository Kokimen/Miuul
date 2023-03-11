import numpy as np
a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])
a*b

np.zeros(10, int)

test = np.random.randint(0, 10, 9)

np.random.normal(10, 4, (3, 4))

test.ndim
test.shape
test.size
test.dtype

test_index = test.reshape(3, 3)  # içeriğini günceller.

test[0:5]

test_index[0, 0]

test_index[:, 2]

test = np.arange(0, 30, 3)

catch = [1, 2, 3]

test[catch]  # fancy index: test array'inin 1, 2 ve 3. elemanını seçer.

a[a < 3]  # np vektörel seviyeden işlem sağlar.

a / 2

np.subtract(a, 5)
np.add(a, 5)
np.mean(a)
np.sum(a)
np.min(a)
np.max(a)
np.var(a)

# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5, 1], [1, 3]])
b = np.array([12, 10])

np.linalg.solve(a, b)  # problem çözme fonksiyonu.