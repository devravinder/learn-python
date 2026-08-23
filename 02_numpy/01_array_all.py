import numpy as np

# this is single file contains all -> see seperate files


print(np.__version__)

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

print("=======================")

array = np.array([
    [['A', 'B', 'C'], ['D', 'E', 'F'], ['G', 'H', 'I']],
    [['J', 'K', 'L'], ['M', 'N', 'O'], ['P', 'Q', 'R']],
    [['S', 'T', 'U'], ['V', 'W', 'X'], ['Y', 'Z', 'a']]
])

print("array", array)
print("array[0][0][0]", array[0][0][0])  # index chaining
print("array[0,0,0]", array[0, 0, 0])  # depth, row, column
print("array[1,1,1]", array[1, 1, 1])

word = array[1, 1, 1] + array[1, 1, 2] + array[2, 0, 1]  # NOT
print("array[1,1,1] + array[1,1,2] + array[2,0,1]", word)

print("===============arr[start:end:step]==============")

arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12],
    [13, 14, 15],
    [16, 17, 18]
])

print("===========row selection arr[start:end:step]=================")
print("arr", arr)

print("arr[1]", arr[1])
print("arr[-1]", arr[-1])

# arr[start:end:step]
print("arr[1:2]", arr[1:2])
print("arr[1:5:2]", arr[1:5:2])  # start from 1st row, till 5th row(excluded) with step 2.  i.e 1st row, 3rd row
# step is like for loop increment


print("arr[::2]", arr[::2])
print("arr[2]", arr[2])

print("all = arr[::1]", arr[::1])
print("all in reverse = arr[::1]", arr[::-1])

print("===========column selection arr[row,column]=================")
print("arr", arr)
print("arr[1,2]", arr[1, 2])
print("all 2nd column elements arr[:,2]", arr[:, 2])
print("2nd last column arr[:,-2]", arr[:, -2])
print("================slicing=arr[row-start:end:step ,column-start:end:step]==============")
print("arr", arr)
print("all rows, columns start-1, end-3 = arr[:,1:3]", arr[:, 1:3])  # arr[row-start:end:step ,column-start:end:step]
print("arr[:, 1:]", arr[:, 1:])
print("arr[:, ::2]", arr[:, ::2])
print("arr[:, 1::2]", arr[:, 1:2])
print("arr[:, ::-1]", arr[:, ::-1])
print("arr[0:2, 0:2]", arr[0:2, 0:2])
print("arr[0:2, 2:]", arr[0:2, 2:])

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

print("=====================aggregate functions: summerize & return single value ==================")
# aggregate functions: summerize & return single value

array = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10]
])

print(np.sum(array))
print(np.max(array))
print(np.min(array))
print(np.argmin(array))  # arg min = min index
print(np.argmax(array))  # arg max = max index
print(np.mean(array))  # mean = avg
print(np.std(array))  # standard deviation
print(np.var(array))  # variation = square of standard deviation

print(np.sum(array, axis=1))  # [15 40] # row wise sum                             # ****

print("=================filtering===============")
ages = np.array([
    [10, 19, 18, 28, 54, 65, 33],
    [1, 17, 27, 34, 87, 69, 77]
])

teenagers = ages[ages < 18]
print(teenagers)
adults = ages[(ages >= 18) & (ages < 65)]
print(adults)

good_people = ages[(ages < 18) | (ages > 65)]
print(good_people)

seniors = ages [ ages > 65]
print(seniors)

evens = ages [ ages %2== 0]
print(evens)

# to preserve the dimension
adults = np.where(ages >18, ages, 0) # condition, array, value_to_fill                   #  ***
print(adults)



print("================random numbers============")

rng = np.random.default_rng() # random number generator

print(rng.integers(low=1, high=100))
print(rng.integers(low=1, high=100, size=3))
print(rng.integers(low=1, high=100, size=(3,2))) # 3-rows, 2 columns


print("================seed============")
rng = np.random.default_rng(seed=1) # seed: to generate same random numbers if run multiple times

print(rng.integers(low=1, high=100))
print(rng.integers(low=1, high=100, size=3))
print(rng.integers(low=1, high=100, size=(3,2))) # 3-rows, 2 columns


print("=================uniform=============")

# old styles, sets random state globally
print(np.random.uniform()) # between 0 and 1
print(np.random.uniform(low=-1, high=1))
print(np.random.uniform(low=-1, high=1, size=3))
print(np.random.uniform(low=-1, high=1, size=(3,2)))

print("===========uniform seed ================")
np.random.seed(seed=1)
print(np.random.uniform(low=-1, high=1, size=(3,2))) # gives same result on running again


print("=================shuffle==================")

rng = np.random.default_rng()
array = np.array([1,2,3,4,5])
rng.shuffle(array)
print(array)

#
fruits = np.array(["🍍","🍎","🍓","🍇","🍉","🍒","🍑", "🥝","🥑"])

fruit = rng.choice(fruits)
print(fruit)

three_fruits = rng.choice(fruits, size=3)
print(three_fruits)

many_fruits = rng.choice(fruits, size=(3,3))
print(many_fruits)


too_many_fruits = rng.choice(fruits, size=(3,3, 3))
print(too_many_fruits)
