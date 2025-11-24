import numpy as np

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
