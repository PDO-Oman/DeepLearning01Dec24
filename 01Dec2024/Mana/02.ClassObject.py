# Class and Object

class Car:
    # Constructor to initialize the object
    def __init__(self, make, model, year):
        self.make = make       # Attribute
        self.model = model     # Attribute
        self.year = year       # Attribute
        
    # Method to display car information
    def display_info(self):
        print(f"{self.year} {self.make} {self.model}")

# Creating an object (instance) of the Car class
my_car = Car("Toyota", "Corolla", 2020)

# Accessing the object's method
my_car.display_info()  # Output: 2020 Toyota Corolla

