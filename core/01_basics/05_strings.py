course = "Python Programming"
print(f"Length: {len(course)}")

'''
The f before a string means f-string → formatted string literal.
It allows you to insert variables or expressions directly inside the string using { }.
'''

print(course[0])
print(course[-1])
print(course[0:6])
print(course[2:6])
print(course[:6])  # 0:6
print(course[:])  # return copy

print(course.lower(), course.upper())

article = """
Hi, I'm Ravinder
Good to see you all
"""

print(article)

# escape char is \
# escape sequences are ", ', \, #, \n (new line)

message = "Hello, \"come\""
print(message)


################################ Functions & Methods #############
course = "    python Programming  "
print(len(course)) # functions
print(course.upper()) # method
print(course.lower())
print(course.title()) # Python Programming 
print(course.strip())
print(len(course.strip()))
print(course.lstrip())
print(course.rstrip())
print(course.find("Pro")) # index
print(course.replace("m", "n"))
print("pro" in course) # boolean
print("swift" not in course)