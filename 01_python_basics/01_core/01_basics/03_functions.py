def greet():
    print("Hello, good morning")
    # this function returns 'None'


greet()



def sum(a, b): # a,b are parameters
    return a+b # return type


f = 2
s = 3
total = sum(f, s) # f,s are args  # # parameters are placeholders for args

print(f, "+", s, "=", total)
print(greet()) # None


print("=================keyword args==============")

def wish(name, message):
    print(f"{message}, {name}")

wish(name="Ravi", message="Good Morning")


print("=============default args============")

# def increament(number, by=1, multiple): # invalid, # only one optional parameter
def increament(number, by=1):
    return number + by

increament(3)
increament(3,2)
increament(3, by=3)

print("================xrags / varaible args=======")

def add(*nums):
    total=0
    for n in nums:
        total+=n
    return total


print(add(1,2,3,4,5))