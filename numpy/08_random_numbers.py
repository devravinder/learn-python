import numpy as np

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
