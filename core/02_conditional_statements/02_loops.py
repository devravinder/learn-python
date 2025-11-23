
# for loop

# Range
for i in range(5):
    print(i)


# list
ages = [1, 2, 40, 12, 5]
for i in ages:
    print("age: ", i)


# while loop

b = -1
while b <= 0:
    arg = input("enter a positive value: ")
    b = int(arg)

print("Given value is: ", b)


print("================pattern===========")
for num in range(1, 5):
  print(num*"*")


print("================range with setp===========")
print("---------step-------------")
for num in range(0, 10, 2):
  print(num)



print("============for & break===========")

numbers = [3, 7, 9, 2, 5]

for n in numbers:
    if n == 2:
        print("Found 2 — stopping loop")
        break
    print("Checking:", n)

print("============for-else===========")

numbers = [3, 7, 9, 5]

for n in numbers:
    if n == 2:
        print("Found 2")
        break
else: # this else will execute...if the for loop not executed 'break'
    print("2 not found in the list")


print("===========seond for-else===========")

numbers = [3, 2, 9, 5]

for n in numbers:
    if n == 2:
        print("Found 2")
        break
else:
    print("2 not found in the list")


print("============while - else=============")
 # this different from for-else (opposite)
 # bcz while (true) then else
x = 1

while x <= 5:
    print("x =", x)
    x += 1
else: # The else block runs only if the while loop finishes normally (NO break used).
    print("Loop finished without break")


print("=============while - else ----second==============")
x = 1

while x <= 5:
    print("x =", x)
    if x == 3:
        break
    x += 1
else: # will not executed
    print("Loop finished")



print("=================nested loops==================")

for i in range(1, 4):
    for j in range(1, 4):
        print(f"i = {i}, j = {j}")



print("==============range===========")
print(type(range(5))) # range type # <class 'range'> & it is iterable


print("=============iteration=================")
text = "Python"

for ch in text:
    print(ch)



text = "Hello"
i = 0

while i < len(text):
    print(text[i])
    i += 1


fruits = ["apple", "banana", "mango"] # tuple

for item in fruits:
    print(item)




print("============enumarate===============")

text = "World"

for index, ch in enumerate(text): # index, value ( not key )
    print(index, ch)




print("============set with enumerate==========")

numbers = {10, 20, 30}

for index, value in enumerate(numbers):
    print(index, value)


print("=============dict iteration==========")
person = {"name": "Ravi", "age": 25, "city": "Hyderabad"}

print("------key------")
for key in person:
    print(key)

print("------value------")


for value in person.values():
    print(value)

print("------key & value------")


for key, value in person.items():
    print(key, "=>", value)

print("------index, key & value------enumerate")


person = {"name": "Ravi", "age": 25, "city": "Hyderabad"}

for index, (key, value) in enumerate(person.items()):
    print(index, key, value)
