x = input("Enter a number: ")  # treated as string

x = int(x)

if x > 2:
    # colon (:) is like an opening curly bracket ({)
    print("Given number is: ", x)
    print("The number ", x, " is greater than ", 2)
    # all the statemments under the same indentation are executed
else:
    print("Given number is: ", x)
    print("The number ", x, " is less than or equal to ", 2)


if x > 2:
    print("Given number is: ", x)
    print("The number ", x, " is greater than ", 2)
elif x < 2:
    print("Given number is: ", x)
    print("The number ", x, " is less than ", 2)
