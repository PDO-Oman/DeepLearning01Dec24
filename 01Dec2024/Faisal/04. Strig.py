# Using f-strings (Python 3.6+)
name = "Faisal"
age = 36
city = "Muscat"

# f-string formatting (simplest and modern way)
print(f"Hello, my name is {name}, I am {age} years old, and I live in {city}.")

# Using str.format() method (older way but still very common)
print("\nUsing str.format() method:")
formatted_string = "Hello, my name is {}, I am {} years old, and I live in {}.".format(name, age, city)
print(formatted_string)

# Using positional and keyword arguments with str.format()
print("\nUsing positional and keyword arguments with str.format():")
formatted_string_pos_kw = "Hello, my name is {0}, I am {1} years old, and I live in {2}.".format(name, age, city)
formatted_string_kw = "Hello, my name is {name}, I am {age} years old, and I live in {city}.".format(name=name, age=age, city=city)
print(formatted_string_pos_kw)
print(formatted_string_kw)

# Using the % operator (older style of formatting)
print("\nUsing % operator (old style formatting):")
print("Hello, my name is %s, I am %d years old, and I live in %s." % (name, age, city))
