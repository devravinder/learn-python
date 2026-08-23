import numpy as np

print("============broadcasting=============")
# arithmetic operations on arrays with arrays — without needing to manually reshape or loop.

# the arrays should have same dimension or one of them should have size 1

array1 = np.array([[1, 2, 3, 4]])
array2 = np.array([[5], [6], [7], [8]])

print(array1.shape)  # (1,4)
print(array2.shape)  # (4,1)
print(array1 * array2)  # not same dimensions, but one has size=1 => (4,4) dimension array

print("=====================")
array1 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8]
])
array2 = np.array([[5], [6], [7], [8]])

print(array1.shape)  # (2,4)
print(array2.shape)  # (4,1)
# print(array1 * array2)  # error, no same dimensions or no size 1 array

print("=====================")

array1 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])  # (4,4)
array2 = np.array([[5], [6], [7], [8]])  # (4,1)

print(array1 * array2)  # result (4,4) # possible rows has same dimensions & columns - one of them has 1 size

print("=====================")

array1 = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])  # (4,4)
array2 = np.array([[5, 1], [6, 2], [7, 3], [8, 2]])  # (4,2)

# print(array1 * array2)  # not possible
print(array1, array2, "(array1 * array2)  # not possible ")

print("========================tables================")
array1 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
array2 = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

print(array1.shape)  # (1,10)
print(array2.shape)  # (10,1)
print(array1 * array2)  # (10, 10)
