a, b = 10, 3

###############
print(a+b)
print(a-b)
print(a*b)
print(a/b)  # 3.3333333333333335
print(a//b)  # for integer = 3                     ****
print(a % b)  # modules / reminder
print(a**b) # exponential a power b


#################
x=3
x+=5 # self assigning
print(x)


print("============comparision============")
# <, >, <=, >=, ==, !=
print(f"a={a}, b={b}")
print(f"a<b:{a<b}, a>b:{a>b}, a<=b:{a<=b}, a>=b:{a>=b}, a==b:{a==b}, a!=b:{a!=b}")

print(10=="10") # False
print(10!="10")

print("---------------------------")
print("bag" > "apple") # True b > a
print("bag"=="BAG") # False
print(f"ord(\"b\")={ord("b")}") # a=97, b=98


print("=====================logical=====================")
# and, or, not


#---- and
age = 20
is_student = True

if age >= 18 and is_student:
    print("You are eligible for student discount")
else:
    print("Not eligible")


#---- or
day = "Saturday"

if day == "Saturday" or day == "Sunday":
    print("Weekend")
else:
    print("Weekday")


#----- not
is_logged_in = False

if not is_logged_in:
    print("Please login first")
else:
    print("Welcome!")


#--------------- short circuit with logical operator
age = 20
has_id = True
is_banned = False

if (age >= 18 and has_id) or not is_banned:
    print("Entry allowed")
else:
    print("Entry denied")


x = 5
print("OK" if (x > 0 and x < 10) or not False else "NO") # this ternary operator



print("=================chaining comparison operators==========")
# like maths
x = 20

if 18 < x <= 21:
    print("Teenager (18 to 21 range)")
else:
    print("Not in the range")
