import numpy as np

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
