# write
file = open("demo.txt", "w")
file.write("Hello Python\n")
file.close()

# read
file = open("demo.txt", "r")
print(file.read())
file.close()
