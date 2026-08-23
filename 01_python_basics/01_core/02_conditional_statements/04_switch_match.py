

day = 3

match day:
    case 1:
        print("Sunday")
    case 2:
        print("Monday")
    case 3:
        print("Tuesday")
    case 4:
        print("Wednesday")
    case _: # default
        print("Invalid day")

print("==============================================")

fruit = "apple"

match fruit:
    case "apple":
        print("Red fruit")
    case "banana":
        print("Yellow fruit")
    case "orange":
        print("Citrus fruit")
    case _:
        print("Unknown fruit")

print("==============================================")


color = "blue"

match color:
    case "red" | "maroon":
        print("Warm color")
    case "blue" | "skyblue":
        print("Cool color")
    case _:
        print("Unknown color")

print("==============================================")
# Pattern matching

user = ("Ravi", 25)

match user:
    case (name, age):
        print(f"Name: {name}, Age: {age}")

print("==============================================")
# with if

num = 10

match num:
    case n if n > 0:
        print("Positive")
    case n if n < 0:
        print("Negative")
    case 0:
        print("Zero")
