import numpy as np

# print(np.__version__)

array = np.array([1, 2, 3, 4])
array = array * 2
print("array", array)
print("type(array)", type(array))  # <class 'numpy.ndarray'> # n-dimensional

###############
print("============dimensions=======")
array = np.array('A')
print("array", array)
print("array.ndim", array.ndim)  # no of dimensions # 0

array = np.array([1, 2, 3, 4])
print("array", array)
print("array.ndim", array.ndim)  # no of dimensions # 1

array = np.array([
    ['A', 'B', 'C'],
    ['D', 'E', 'F'],
    ['G', 'H', 'I']
])
print("array", array)
print("array.ndim", array.ndim)  # no of dimensions # 2

array = np.array([
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']],
    [['J', 'K', 'L'], ['M', 'N', 'O'], ['P', 'Q', 'R']],
    [['S', 'T', 'U'], ['V', 'W', 'X'], ['Y', 'Z', 'a']],
    # [['S','T','U'],['V','W','X'],['Y','Z']], # this gives error...all should have equal no of elements
])
print("array", array)
print("array.ndim", array.ndim)  # no of dimensions # 3

print("array.shape", array.shape)  # (3,3,3) (depth, rows, columns)

array = np.array([
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']],
    [['J', 'K', 'L'], ['M', 'N', 'O'], ['P', 'Q', 'R']]
])

print("array", array)
print("array.ndim", array.ndim)  # no of dimensions # 3

print("array.shape", array.shape)  # (2,3,3) (depth, rows, columns)
