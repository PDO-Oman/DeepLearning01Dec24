# List - Mutable collection
my_list = [1, 2, 3, 4, 5]
print("Original List:", my_list)

# Add a new element to the list
my_list.append(6)
print("List after appending 6:", my_list)

# Remove an element from the list
my_list.remove(3)
print("List after removing 3:", my_list)

# Accessing elements in the list
print("First element in the list:", my_list[0])

# Tuple - Immutable collection
my_tuple = (10, 20, 30, 40, 50)
print("\nOriginal Tuple:", my_tuple)

# Accessing elements in the tuple
print("Third element in the tuple:", my_tuple[2])

# Dictionary - Collection of key-value pairs
my_dict = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}
print("\nOriginal Dictionary:", my_dict)

# Accessing values using keys
print("Name from the dictionary:", my_dict["name"])

# Adding a new key-value pair to the dictionary
my_dict["occupation"] = "Engineer"
print("Dictionary after adding occupation:", my_dict)

# Removing a key-value pair from the dictionary
del my_dict["age"]
print("Dictionary after removing age:", my_dict)

# Iterating through a dictionary
print("\nIterating through dictionary keys and values:")
for key, value in my_dict.items():
    print(f"{key}: {value}")
