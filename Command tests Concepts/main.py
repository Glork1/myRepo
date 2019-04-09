
'''
LIST COMPREHENSIONS

[ expression for item in list if conditional ]
*result*  = [*transform*    *iteration*         *filter*]

'''

a = [0,2,3,5,6,7,8,9,4,5]

b = [i*i for i in a if (i%2)==0] # [0,4,36,64,16]
c = [i for i in range(10)] # [0,1,2,3,4,5,6,7,8,9]

squares_without = []

for i in range(10):
    squares_without.append(i*i)


squares_with = [i**2 for i in range(10)]

listOfWords = ["this","is","a","list","of","words"]

items = [ word[0] for word in listOfWords ]
[x.lower() for x in ["A","B","C"]]

# Create a function and name it double:
def double(x):
  return x*2

[double(x) for x in range(10)]



'''
ITERATORS

Iterators are objects that can be iterated upon.
An object is called iterable if we can get an iterator from it:  e.g. list, tuple, string... are iterables 

Technically speaking, Python iterator object must implement two special methods, 
__iter__() and __next__(), 
collectively called the iterator protocol.

We use the next() function to manually iterate through all the items of an iterator.
'''

# define a list
my_list_to_iterate = [4, 7, 0, 3]

# get an iterator using iter()
my_iter = iter(my_list_to_iterate)

## iterate through it using next() 

#prints 4
print(next(my_iter))

#prints 7
print(next(my_iter))

s = "hello"

'''
GENERATORS

There is a lot of overhead in building an iterator in Python; we have to implement a class with
 __iter__() and __next__() method, keep track of internal states, raise StopIteration 
when there was no values to be returned etc.

This is both lengthy and counter intuitive. Generator comes into rescue in such situations.

Python generators are a simple way of creating iterators. 
All the overhead we mentioned above are automatically handled by generators in Python.

Simply speaking, a generator is a function that returns an object (iterator) 
which we can iterate over (one value at a time).

How to create a generator in Python?
It is as easy as defining a normal function with yield statement instead of a return statement


Differences between Generator function and a Normal function

Here is how a generator function differs from a normal function.

    - Generator function contains one or more yield statement.
    - When called, it returns an object (iterator) but does not start execution immediately.
    - Methods like __iter__() and __next__() are implemented automatically. So we can iterate through the items using next().
    - Once the function yields, the function is paused and the control is transferred to the caller.
    - Local variables and their states are remembered between successive calls.
    - Finally, when the function terminates, StopIteration is raised automatically on further calls.

So:
    
Generator functions allow you to declare a function that behaves like an iterator.
They allow programmers to make an iterator in a fast, easy, and clean way.
'''

# A simple generator function
def my_gen():
    n = 1
    print('This is printed first')
    # Generator function contains yield statements
    yield n

    n += 1
    print('This is printed second')
    yield n

    n += 1
    print('This is printed at last')
    yield n


# It returns an object but does not start execution immediately.
a = my_gen()
# We can iterate through the items using next().
next(a) #This is printed first
#1
# Once the function yields, the function is paused and the control is transferred to the caller.

# Local variables and theirs states are remembered between successive calls.
next(a)#This is printed second
#2

next(a)#This is printed at last
#3

# One interesting thing to note in the above example is that, the value of the variable n is remembered between each call.
# Unlike normal functions, the local variables are not destroyed when the function yields. 
# One final thing to note is that we can use generators with for loops directly.

# Using for loop
# for item in my_gen():
#   print(item)  

''' MAP, REDUCE, FILTER, ITER, RANGE, XRANGE (named as functional programming)'''


''' MAP 

map(function_to_apply, list_of_inputs) # similar to list_comprehensions
be careful, not to ommit LIST in front of map to create the list object

instead of:
'''    
items = [1, 2, 3, 4, 5]
squared = []
for i in items:
    squared.append(i**2)

#just do:
    
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items)) 

# here above: less readable than list_comprehension
# see also: LIST(map) so creates  a list and MAP(intuitiv)


# Most of the times we use lambdas with map so I did the same. 
# Instead of a list of inputs we can even have a list of functions!

def multiply(x):
    return (x*x)
def add(x):
    return (x+x)

funcs = [multiply, add]

#for i in range(5):
    #value = list(map(lambda x: x(i), funcs))
    #print(value)

# Output:
# [0, 0]
# [1, 2]
# [4, 4]
# [9, 6]
# [16, 8]

'''

FILTER

filter CREATES a LIST of elements for which a function returns true. Here is a short and concise example:

'''

number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))# intuitive: first filter then create a list out of it
print(less_than_zero) # Output: [-5, -4, -3, -2, -1]


''' 

REDUCE

Reduce is a really useful function for performing some computation on a list and returning the result. 
It applies a rolling computation to sequential pairs of values in a list. 
For example, if you wanted to compute the product of a list of integers.

So the normal way you might go about doing this task in python is using a basic for loop:

'''
product = 1
t_list = [1, 2, 3, 4]
for num in t_list:
    product = product * num

# product = 24
# Now let’s try it with reduce:

from functools import reduce
product = reduce((lambda x, y: x * y), [1, 2, 3, 4])

''' RANGE AND XRANGE

No more difference between the two since Python 3
Before there was a difference in terms of memory.
For example, if we were to create a huge list, xrange is the one to use because range used to create a static list 


'''


'''

DECORATORS

Function decorators are simply wrappers to existing functions
A function that takes another function as an argument, generates a new function,
augmenting the work of the original function, and returning the generated function so we can use it anywhere.

'''
# it's syntax sugaring
# time_dec is a decorator
# it TAKES a FUNCTION as an ARGUMENT and GENERATES/RETURNS a new function (augmenting the work of the original function)
# and it returns this new function
import time

def time_dec(func):
    def wrapper(*arg): #*args in order to accept any arbitrary number of arguments and keywords arguments
      t = time.clock() # or time.time() for mesuring the time
      res = func(*arg)
      print (func.__name__, time.clock()-t)
      return res
    return wrapper


@time_dec
def factorielle(n):
    res = 1
    for i in range(1,n):
        res = res*i
    return res

print(factorielle(10))

def my_dec(func):
    def wrapper(*args):
        print(func.__name__)
    return wrapper

@my_dec
def carre(n):
    return n*n

print(carre(2))



import itertools as it

factorielle_list = it.chain([[factorielle(i),i*i] for i in range(30) if i%2==0])

generatorobj = (i**i for i in range(1, 5))
print(generatorobj) # generator object
print(next(generatorobj)) # 1
print(next(generatorobj)) # 4
print(next(generatorobj)) # 3




