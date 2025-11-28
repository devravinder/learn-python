
##################################### Text #############################################
name="Ram" # text data type or string
village='Hyd' # this is also string

# we can use any of the quotes double ("")  or single ('')

# recoomnded (""),  because used in many laguages

print(name)
print(village)



# data stored memory pointer is called variable: eg: name, village


########################################## Number ########################################

# in python numbers are two types
age=22 # int number
height=1.65 # float number


print(age)
print(height)


########################################## Boolean  ########################################

is_adult=True
is_govt_employee=False

print(is_adult)
print(is_govt_employee)



########################################## Checking data types  ##################################

print("---------------------")

print("name type ", type(name))
print("age type ", type(age))
print("height type ", type(height))

print("is_adult type ", type(is_adult))
print("is_govt_employee type ", type(is_govt_employee))



################################################## Printing #########################################
# printing is just to see the data / to chcek


print(name)  # only variable

print("name is ", name) # variable with some text

print(name, age, height) #  multiple variables

print("name: ", name, "age: ", age, "height: ", height) # multiple variables with text 



# prefix with 'f'
print(f"name is {name} age {age} & height is {height}") # formatted text # no comma only substitution