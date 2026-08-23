arg = input("Enter a number: ")
x = int(arg)
print("int:", x, type(x))


arg = input("Enter a float: ")
x = float(arg)
print("float:", x, type(x))

arg = input("Enter a bool: ")
x = bool(arg)
print("bool:", x , type(x))

print("===========Important================")
arg="False"
print(f"bool({arg}) is {bool(arg)}")

arg = str(x) 
print("string:", arg, type(arg))


# Falsy values: "", 0, None
# Note: bool("False") is true , bcz it is not empty string