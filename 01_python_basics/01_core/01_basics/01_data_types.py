
'''
Primitives:
  - Number
      - int, float, complex
  - Strings
  - boolean

Object(reference):
  - list
  - tuple
  - set
  - dictionary
'''

i = 2  # int
fl = 3.2 # float
c = 2 + 3j # complex = 2 + 3i
s = "Hello"  # string
t = True  # boolean
f = False

if t:
    print("The number is :", i)

if not f: # not=!
    print("The string is :",s)


print("complex", c)
###################
name = "Ravi"
age = 25
height = 5.9
is_married = False
skills = ["Python", "JavaScript"] # list
person = {"name": name, "age": age} # dictionary

print(type(name), type(age), type(height), type(skills))
