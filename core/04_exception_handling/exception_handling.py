try:
    num = int(input("Enter number: "))
    print("Square:", num * num)
except ValueError:
    print("Invalid number")
finally:
    print("Go to another")


print("=============")
try:
    num = int(input("Enter number: "))
    print("Square:", num * num)
except ValueError as e:
    print("Invalid number")
    print("Error info:", e)
finally:
    print("Program done")
