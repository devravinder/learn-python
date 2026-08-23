from colorama import Fore, Style


def calculator():
    print("Basic Calculator (+, -, *, /)")

    a = float(input("Enter first number: "))
    b = float(input("Enter second number: "))
    op = input("Enter operator (+ - * /): ")

    result = None

    if op == "+":
        result = a + b
    elif op == "-":
        result = a - b
    elif op == "*":
        result = a * b
    elif op == "/":
        if b == 0:
            print(Fore.RED + "Error: Division by zero!" + Style.RESET_ALL)
            return
        result = a / b
    else:
        print(Fore.RED + "Invalid operator!" + Style.RESET_ALL)
        return

    print(Fore.GREEN + f"Result = {result}" + Style.RESET_ALL)


if __name__ == "__main__":
    calculator()
