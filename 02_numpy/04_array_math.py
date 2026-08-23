import numpy as np

print("===============scalar arithmetic==========")
# scalar = linear
# scalar operation will be applied on each element
array = np.array([1, 2, 3, 4, 5])
print(array)
print(array + 1)
print(array - 2)
print(array * 3)
print(array / 4)
print(array // 2)
print(array % 2)
print(array ** 2)

print("===============vector math functions============")
array = np.array([1, 2, 3, 4, 5])
print(array)
print(np.square(array))
print(np.sqrt(array))
array = np.array([1.03, 1.41421356, 1.73205081, 2.414, 4.23606798])
print(np.round(array))
print(np.ceil(array))
print(np.floor(array))
print(np.pi)
array = np.array([1, 2, 3, 4, 5])
print(np.power(array, 2))

## EXERCISE
radii = np.array([1, 2, 3, 4])

# area = Pi * r^2
print(np.pi * radii ** 2)

print("=============element wise arithmetic==========")
array1 = np.array([1, 2, 3])
arra2 = np.array([4, 5, 6])
print(array1 + arra2)
print(array1 - arra2)
print(array1 * arra2)
print(array1 / arra2)
print(array1 ** arra2)

print("=============comparison operator===========")
scores = np.array([56, 36, 23, 40])
print(scores > 36)  # [ True False False  True]
print(scores <= 35)  # [False False  True False]
print(scores == 40)  # [False False False  True]
scores[scores < 35] = 0  # fail  # self assign
print(scores)  # [56 36  0 40]
