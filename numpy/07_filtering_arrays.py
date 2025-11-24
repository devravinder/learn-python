import numpy as np

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
