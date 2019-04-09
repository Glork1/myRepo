import time

def pretty_decorator(f):
    def wrapper(*args):
        print("*********")
        f(*args)
        #print("*********")
    return wrapper

def timeToExecute(f):
    def wrapper(*args):
        start = time.clock()
        res = f(*args)
        end = time.clock()
        print("Execution time:" + str(end-start))
    return wrapper
    
        
class Shape:
    def imprime(self):
        print("This is a shape.")
    def setbase(self,b):
        self.b = b

class Cercle(Shape):
    def setx(self,x):
        self.x = x

    def aire(self,x):
        return 3.14*x*x

    @pretty_decorator    
    def imprime(self,x):
        Shape.imprime(self)
        print("In particular a circle. Aire = " + str(Cercle.aire(self,x)))
        
class Rectangle(Shape):
    def setxy(self,x,y):
        self.x = x
        self.y = y
    
    def aire(self,x,y):
        return x*y
    
    @pretty_decorator    
    def imprime(self,x,y):
        Shape.imprime(self)
        print("In particular a rectangle. Aire = "+ str(Rectangle.aire(self,x,y)))
    
        
class Carre(Rectangle):
    
    def setx(self,x):
        self.x = x

    def aire(self,x):
        return x*x
    
    @timeToExecute
    @pretty_decorator    
    def imprime(self,x):
        Shape.imprime(self)
        print("In particular a square. Aire = "+ str(Carre.aire(self,x)))

s = Shape()
s.imprime()

ce = Cercle()
ce.imprime(10)

r = Rectangle()
r.imprime(2,3)

ca = Carre()
ca.imprime(5)

print(vars(s)) # {}
print(vars(ce)) # {}
print(vars(ca)) # {}
print(vars(r)) # {}
print(vars(Shape))
'''
{'__module__': '__main__', 
'imprime': <function Shape.imprime at 0x000000004D7BC950>, 
'__dict__': <attribute '__dict__' of 'Shape' objects>, 
'__weakref__': <attribute '__weakref__' of 'Shape' objects>, 
'__doc__': None}
'''

print(vars(Carre))
'''
{'__module__': '__main__', 
'setx': <function Carre.setx at 0x000000004D7BCE18>,
'aire': <function Carre.aire at 0x000000004D7BCEA0>, 
'imprime': <function timeToExecute.<locals>.wrapper at 0x000000004D7DE048>, 
'__doc__': None}
'''
Shape.y = 10 # We change the definition of the class Shape, by creating a member variable y
print(vars(Shape))
'''
{'__module__': '__main__', 
'imprime': <function Shape.imprime at 0x000000004D7BC950>, 
'__dict__': <attribute '__dict__' of 'Shape' objects>, 
'__weakref__': <attribute '__weakref__' of 'Shape' objects>, 
'__doc__': None, 
'y': 10}
'''

print(vars(Carre))
'''
{'__module__': '__main__', 
'setx': <function Carre.setx at 0x000000004D7BCE18>, 
'aire': <function Carre.aire at 0x000000004D7BCEA0>, 
'imprime': <function timeToExecute.<locals>.wrapper at 0x000000004D7DE048>, 
'__doc__': None}
'''

shape_test = Shape()
print(vars(shape_test)) # {}
'''
At first, this output makes no sense. We just sa that g had the member y, so why isn't it in the member dictionary ?
We actually put "y" in the class definition Shape, not g (see above the result of vars(Shape))
and there we have all the members of the class Shape definition.
When Pyton checks for shape_test.member, 
it first checks shape_test vars dictionary for "member",
then it checks Shape vars dictionary.
If we create a new member of shape_test, it will be added to shape_test's dictionary but not Shape's dictionary
'''

shape_test.setbase(18)
print(vars(shape_test)) # {'b': 18}

''' ---> Dynamic Class Structure: the members of a Python class can change during runtime, unlike C++ <---'''
''' i.e. we can change the definition of the Shape class during program execution '''

shape_test_bis = Shape()
print(vars(shape_test_bis)) # {}
print(shape_test_bis.y) #10
''' Note that shape_test_bis.y will also be 10, as Python won't find "y" in vars(shape_test_bis) [the dictionary of shape_test_bis] '''
''' so it will get the value of "y" from vars(Shape) '''

'''
Some may have also noticed that the methods in Foo appear in the class dictionary along with the x and y. 
If you remember from the section on lambda functions, we can treat functions just like variables. 
This means that we can assign methods to a class during runtime in the same way we assigned variables. 
If you do this, though, remember that if we call a method of a class instance, 
the first parameter passed to the method will always be the class instance itself. 
'''
                               


'''
"__init__" is a reseved method in python classes. It is known as a constructor in object oriented concepts. 
This method called when an object is created from the class and it allow the class to initialize the attributes of a class.
'''
              
class A:
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    def imprime(self):
        print("x = " + str(self.x) )
        print("y = " + str(self.y) )
        
my_a = A(2,3)
print(vars(my_a)) #{'x': 2, 'y': 3}

'''
Note: 
    "my_a" is the representation of the object outside of the class 
and 
    "self"  is the representation of the object inside the class.
'''

''' @property '''


''' An Example To Begin With '''

class Celsius:
    def __init__(self, temperature = 0):
        self.temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

# create new object
man = Celsius()
print(vars(man)) #{'temperature': 0}

# set temperature
man.temperature = 37

# get temperature
print(man.temperature) # 37

# get degrees Fahrenheit
print(man.to_fahrenheit()) #98.60000000000001

'''
Whenever we assign or retrieve any object attribute like temperature, as show above, 
Python searches it in the object's __dict__ dictionary.

print(man.__dict__)  #{'temperature': 37}

Therefore, man.temperature internally becomes man.__dict__['temperature'].

One fateful day, a trusted client came to us and suggested that temperatures cannot go below -273 degree Celsius 
(students of thermodynamics might argue that it's actually -273.15), 
also called the absolute zero. He further asked us to implement this value constraint.

-> Using Getters and Setters <-

An obvious solution to the above constraint will be to hide the attribute temperature (make it private) 
and define new getter and setter interfaces to manipulate it. This can be done as follows.
'''

class Celsius:
    def __init__(self, temperature = 0):
        self.set_temperature(temperature)

    def to_fahrenheit(self):
        return (self.get_temperature() * 1.8) + 32

    # new update
    def get_temperature(self):
        return self._temperature

    def set_temperature(self, value):
        if value < -273:
            raise ValueError("Temperature below -273 is not possible")
        self._temperature = value

'''
We can see above that new methods get_temperature() and set_temperature() were defined and furthermore, 
temperature was replaced with _temperature. 
An underscore (_) at the beginning is used to denote private variables in Python.
'''

#c = Celsius(-277) ,- will raise an error

'''
Please note that private variables DON'T EXIST in Python. 
There are ARE SIMPLY NORMS TO BE FOLLOWED. The language itself does mot apply any restrictions.
'''
c = Celsius(25)
c.set_temperature(10)
c._temperature = -300 # !!!
print(c.get_temperature()) # -300

'''     
But this is not of great concern. 
The big problem with the above update is that, 
all the clients who implemented our previous class in their program have to modify their code 
from obj.temperature to obj.get_temperature() and all assignments like obj.temperature = val to obj.set_temperature(val).
This refactoring can cause headaches to the clients with hundreds of thousands of lines of codes.     
All in all, our new update was WAS NOT BACKWARD COMPATIBLE. This is where property comes to rescue.
The pythonic way to deal with the above problem is to use property. Here is how we could have achieved it:

'''


class Celsius:
    def __init__(self, temperature = 0):
        self.temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    def get_temperature(self):
        print("Getting value")
        return self._temperature

    def set_temperature(self, value):
        if value < -273:
            raise ValueError("Temperature below -273 is not possible")
        print("Setting value")
        self._temperature = value

    temperature = property(get_temperature,set_temperature)



c = Celsius() #  Setting value


'''
We added a print() function inside get_temperature() and set_temperature() to clearly observe that they are being executed.
The last line of the code, makes a property object temperature. 
Simply put, property attaches some code (get_temperature and set_temperature) to the member attribute accesses (temperature).

--------------------------------------------------------------------------------------------------------------------------------------
| Any code that retrieves the value of temperature will automatically call get_temperature() instead of a dictionary (__dict__) look-up.
| Similarly, any code that assigns a value to temperature will automatically call set_temperature(). 
--------------------------------------------------------------------------------------------------------------------------------------

This is one cool feature in Python.
We can see above that set_temperature() was called even when we created an object.
Can you guess why?
The reason is that when an object is created, __init__() method gets called. 
This method has the line self.temperature = temperature. This assignment automatically called set_temperature().
'''


c.temperature # Getting value


'''
By using property, we can see that, we modified our class and implemented the value constraint 
without any change required to the client code. 
Thus our implementation was backward compatible and everybody is happy.

--------------------------------------------------------------------------------------------------------
|   Finally note that, the actual temperature value is stored in the private variable _temperature.     |
|   The attribute temperature IS A PROPERTY OBJECT WHICH PROVIDES AN INTERFACE TO THIS PRIVATE VARIABLE |
--------------------------------------------------------------------------------------------------------

A property object has three methods, getter(), setter(), and deleter() to specify fget, fset and fdel at a later point. 
This means, the line: temperature = property(get_temperature,set_temperature)
could have been broken down as

# make empty property
temperature = property()
# assign fget
temperature = temperature.getter(get_temperature)
# assign fset
temperature = temperature.setter(set_temperature)


Programmers familiar with decorators in Python can recognize that the above construct can be implemented as decorators.
We can further go on and not define names get_temperature and set_temperature as they are unnecessary and pollute the class namespace.
For this, we reuse the name temperature while defining our getter and setter functions. This is how it can be done.
'''

class Celsius:
    def __init__(self, temperature = 0):
        self._temperature = temperature

    def to_fahrenheit(self):
        return (self.temperature * 1.8) + 32

    @property
    def temperature(self):
        print("Getting value")
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if value < -273:
            raise ValueError("Temperature below -273 is not possible")
        print("Setting value")
        self._temperature = value

d = Celsius()
d.temperature = 20 # Setting value


'''
In summary:
    
@property
def x(self):
    return self._x

is equivalent to:

def getx(self):
    return self._x
x = property(getx)

'''



''' Static method '''

'''
Static methods in Python are just like their counterparts in C++ or Java. Static methods have no "self" argument 
and don't require you to instantiate the class before using them. They can be defined using staticmethod() 
'''

class StaticSpam(object):
    def StaticNoSpam():
        print ("Blablabla")
    NoSpam = staticmethod(StaticNoSpam)
    
StaticSpam.NoSpam() # Blablabla

''' They can also be defined using the function decorator @staticmethod.  '''

class StaticSpam(object):
    @staticmethod
    def StaticNoSpam():
        print ("Blablabla")
    
StaticSpam.StaticNoSpam() # Blablabla


''' ---------> SPECIAL METHODS <--------- '''


''' ---------> INITIALIZATION AND DELETION <--------- '''

''' __init__ '''

class  A:
    def __init__(self,x):
        self.x = x
    def imprime(self):
        print(self.x)
a = A(2)
a.imprime() #2

''' __del__  '''

''' __new__ '''

''' ---------> REPRESENTATION <--------- '''

''' __str__  and __repr

CONVERTING an OBJECT to a STRING, as with the print statement or with the str() conversion function,
can be overridden by overriding __str__. Usually, __str__ returns a formatted version of the objects content.

__str__ is the built-in function in python, used for string representation of object.
__repr__ is another built-in which is similar to __str__.

Both of them can be OVERRIDEN for any class and there are minor differences.
If both are defined, function defined in __str__ is used
If __repr__ is defined but not __str__, logically __repr__ is used
'''

class Student:
    def __init__(self,identity,name):
        self.identity = identity
        self.name = name
    def __str__(self):
        return "Identity student: " + str(self.identity) + " / Name: " + self.name # The object "self" is converted into a string

s = Student(2536,"Sam")
print(s)  # Identity student: 2536 / Name: Sam



''' ---------> ATTRIBUTES <--------- '''

'''
Function 	Indirect form 	Direct Form
__getattr__ 	getattr(A, B) 	A.B
__setattr__ 	setattr(A, B, C) 	A.B = C
__delattr__ 	delattr(A, B) 	del A.B 

'''

''' __setattr__ 

This is the function which is in charge of setting attributes of a class. 
It is provided with the name and value of the variables being assigned. 
Each class, of course, comes with a default __setattr__ which simply sets the value of the variable, 
but we can override it.

It is CALLED when an attribute assignment is attempted.

'''

class Unchangable:
    def __setattr__(self, name, value):
        print("Nice try")

u = Unchangable()
u.x = 9 # Remember we can hre try to modify the member variables outside of the class

''' __getattr__ 
Similar to __setattr_ but this function is called when trying to access a class member, and the default simply returns the value '''

''' __delattr__ 
This function is called to delete an attribute
'''

class Permanent:
    def __delattr__(self, name):
        print(name, "cannot be deleted")

p = Permanent()
p.x = 9
del p.x #x cannot be deleted

''' ---------> OPERATOR OVERLOADING <--------- 
Operator overloading allows us to use the built-in Python syntax and operators to call functions which we define. '''

''' ---------> BINARY OPERATORS <--------- 
If a class HAS the __add__ function, we can use the '+' operator to add instances of the class. 
This will call __add__ with the two instances of the class passed as parameters, 
and the return value will be the result of the addition. 
'''

class FakeNumber:
    n = 5
    def __add__(A,B):
        return A.n + B.n

c = FakeNumber()
d = FakeNumber()
d.n = 7
print(c + d) # 12 

class Account:
    def __init__(self,x):
        self.x = x
    def __add__(self,other):
        return self.x + other.x

a = Account(10)
b = Account(20)
print(a+b) # 30


''' ---------> ITEM OPERATORS <--------- 

It is also possible in Python to override the indexing and slicing operators. 
This allows us to use the class[i] and class[a:b] syntax on our own objects.

The simplest form of item operator is __getitem__. 
This takes as a parameter the instance of the class, then the value of the index. '''

class DoubleItem:
    def __getitem__(self,index):
        return index * 2

double_item = DoubleItem()
print(double_item['a']) # aa

''' ---------> PROGRAMMING PRACTICES <--------- '''

''' EMCAPSULATION 

Since all python members of a python class are accessible by functions/methods outside the class, 
there is no way to enforce encapsulation short of overriding __getattr__, __setattr__ and __delattr__. 
General practice, however, is for the creator of a class or module to simply trust 
that users will use only the intended interface and avoid limiting access to the workings of the module 
for the sake of users who do need to access it. 
When using parts of a class or module other than the intended interface, 
keep in mind that the those parts may change in later versions of the module, 
and you may even cause errors or undefined behaviors in the module.since encapsulation is private. 
'''

''' DOC STRINGS
When defining a class, it is convention to document the class using a string literal at the start of the class definition. 
This string will then be placed in the __doc__ attribute of the class definition.
'''
class Example:
    """This is a docstring"""
    def imprime(self):
        """ This method is documented too """
        print("coucou")

e = Example()
print(e.__doc__) # This is a docstring

''' ADDING METHODS AT RUNTIME '''

''' TO A CLASS

It is fairly easy to add methods to a class at runtime. 
Lets assume that we have a class called Spam and a function cook. 
We want to be able to use the function cook on all instances of the class Spam:
'''

class Spam:
  def __init__(self):
    self.myeggs = 5

def cook(self):
  print("cooking %s eggs" % self.myeggs)

Spam.cook = cook   # ADD the function to the class Spam
eggs = Spam()      # NOW create a new instance of Spam
eggs.cook()        # and we are ready to cook! Displays: cooking 5 eggs

''' TO AN INSTANCE OF A CLASS
It is a bit more tricky to add methods to an instance of a class that has already been created. 
Lets assume again that we have a class called Spam and we have already created eggs. 
But then we notice that we wanted to cook those eggs, 
but we do not want to create a new instance but rather use the already created one.

'''
'''
class Spam:
  def __init__(self):
    self.myeggs = 5

eggs = Spam() # The instance is ALREADY created

def cook(self):
  print("cooking %s eggs" % self.myeggs)

import types
f = types.MethodType(cook, eggs, Spam)

eggs.cook = f
eggs.cook()
'''

