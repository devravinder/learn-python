
lst = [10,20,30]
tpl = (1,2,3) # fixed
dct = {"name":"Ravi","age":25} # dictonary
st = {1,2,3,3} # set

lst.append(40)
print(lst, tpl, dct["name"], st)


print("=========list==========")
fruits = ["apple", "banana", "orange", "banana"]
fruits[1] = "grape"   # allowed
print(fruits)


print("============tuple========")
colors = ("red", "green", "blue", "green")
# colors[1] = "yellow"  ❌ error (cannot modify)
print(colors)


print("==============set==========")
numbers = {1, 2, 3, 3, 4}
print(numbers)  # output: {1, 2, 3, 4}  (removes duplicate)
print(4 in numbers)   # True




print("================Dictionary=============")
person = {"name": "Ravi", "age": 25}
person["age"] = 26     # update value
print(person["name"])
