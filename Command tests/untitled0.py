import numpy as np

# LIST
# array:  collection of elements of a single data type, eg. array of int, array of string.
# In Python no native array data structure, so use of Python lists instead of an array)
# To create real arrays in Python, use NumPy's array data structure. For mathematical problems, NumPy Array is more efficient
# Unlike arrays, a single list can store elements of any data type and does everything an array does.
# We can store an integer, a float and a string inside the same list. So, it is more flexible to work with.
# [10, 20, 30, 40, 50 ] is an example of what an array would look like in Python, but it is actually a list.

myList = [1,2,3,"hello"]
print(myList[0])
print(myList[-1])
print(len(myList))
myList.append("hu")
del myList[0]
myList.remove("hello")
myList.remove("hu")
myList[0]=1
myList += [5,7]
repeat = ["Hi"]
repeat = repeat * 5
mySlice = myList[1:3] # careful take elements from index 1 inclusive to index 3 exclusive
mySlice  = myList[-4:] # all elements after 2nd position, equivalent to myList[1:]

myList.append([9,11]) # [1,3,5,7,[9,7]]
myList.extend([13,15]) # [1,3,5,7,[9,7],13,15], list concatenation: append all the elements contained in the list rather than the list itself. Similar to the operator +=, which is faster, cause former needs a function call
myList.insert(len(myList),17) # [1,3,5,7,[9,7],13,15,17] st argument is the index of the element before whih to insert
myList.pop(-1) # [1,3,5,7,[9,7],13,15]
print(myList.count(17)) # 0
myList.reverse() # [15,13,[9,7],7,5,3,1]
myList.sort() # [1,3,5,7,13,15,[9,7]] in O(nlog(n)) ....sort(reverse=True) renvoie la liste dans l ordre decroisssant
print(myList.index(15)) # returns the lowest index in list that obj appears, otherwise raises an exception

multd = [[1,2], [3,4], [5,6], [7,8]]

# NUMPY ARRAY

a = np.array([1,2,3.0]) # and upcasting
a_mult = np.array([[1, 2], [3, 4]])

## all
zero_ret = np.zeros(5)
print(np.all(zero_ret == 0)) # returns True if all elements evaluate to True.
print(np.all(i < 0 for i in zero_ret)) # returns False

## arg todo
print(np.argmin(a)) # return the indice of the minumum values along an axis

## astype
x = np.array([1,2,2.5]) # upcasting, numpy always manipulates homogeneous data for his math operations
print(x) # [1,2,2.5]
x = x.astype(int) # copy of the array, cast to a specified type
print(x) # [1,2,2]

## choose


## clip 
xx = np.arange(10) # [0,1,2,3,4,5,6,7,8,9]
trunc_x = np.clip(xx,1,8) # [1,1,2,3,4,5,6,7,8,8] clip (limit) the values in an array. Given an interval, values outside the interval are clipped to the interval edges. For example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values larger than 1 become 1.
yy = np.arange(10)
np.clip(yy,1,8,out=yy) ## yy is modified
zz = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.clip(zz, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8,out=zz) ## [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

## compress
ex_comp = np.array([[1, 2], [3, 4], [5, 6]])
new_ex_ccomp = np.compress([0,1],ex_comp,axis=0) # returns [3,4]. [0,1] or [False,True] is the array of boolean that selects which entrued to return. If len(condition) is less than the size of a along the given axis, then output is truncated
print(np.compress([False,True],np.array([1,2,3]))) # returns 2

## copy
ff = np.array([1,2,3])
gg = ff.copy() # returns a copy of the array


## reshape
# numpy allow us to give one of new shape parameter as -1 (eg: (2,-1) or (-1,3) but not (-1, -1))
# It simply means that it is an unknown dimension and we want numpy to figure it out.
rr = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])
print(rr.shape) # (3L,4L)

# Trying to reshape with (-1) . Result new shape is (12,) and is compatible with original shape (3,4)
r_colo = rr.reshape(-1) #[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

# New shape as (-1, 2). row unknown, column 2. we get result new shape as (6, 2)
r_bis = rr.reshape(-1,2) #array([[ 1,  2],[ 3,  4],[ 5,  6],[ 7,  8],[ 9, 10],[11, 12]]

# New shape as (1,-1). i.e, row is 1, column unknown. we get result new shape as (1, 12)
r_ter = rr.reshape(1,-1) #array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]]

# New shape (2, -1). Row 2, column unknown. we get result new shape as (2,6)
r_quat = rr.reshape(2, -1) #array([[ 1,  2,  3,  4,  5,  6],[ 7,  8,  9, 10, 11, 12]])

# and finally, if we try to provide both dimension as unknown i.e new shape as (-1,-1). It will throw an error




##where
awhere = np.array([1,2,3,4,5,6,7])
awhere = np.where(awhere>5,"Yes","No")

blist = np.array([1,2,3,4,5,6,7])
#blist = ["+" if int(i)>5 else "-" for i in awhere] TRY

# arange
ar = np.arange(3) # array([0, 1, 2])
br = np.arange(3,7) # array([3, 4, 5, 6])
cr = np.arange(3,7,2) # array([3, 5])

# numpy.random
from numpy import random as rd

ary = list(range(10))
rd.choice(ary, size=8, replace=False) # array([0, 5, 9, 8, 2, 1, 6, 3])  # no repeated elements
rd.choice(ary, size=8, replace=True) # array([4, 9, 8, 5, 4, 1, 1, 9])  # elements may be repeated

# numpy.random.rand(n,1) (uniform)
xrandom = np.random.rand(n,1)

# numpy.random.normal (gaussian)  (numpy.random.random (uniform))
mu, sigma = 0, 0.5 # mean and standard deviation
noise = np.random.normal(mu, sigma, n).reshape(-1,1)

# numpy.random.standard_normal(size=None) : draw samples from a standard Normal distribution (mean=0, stdev=1)
# size: Output shape. If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn. Default is None, in which case a single value is returned.
# out : float or ndarray
noise_bis = np.random.standard_normal(8000)

# numpy.random.multivariate_normal: to draw correlated gaussian variables (ie X=[X1,...,Xn] is a gaussian vector with mean m=[...] and covariance matrix Sigma = [...]) see Chap8_bagging
mean_mult = (1, 2)
cov_mult = [[1, 0], [0, 1]]
x = np.random.multivariate_normal(mean_mult, cov_mult, (3, 3)) # Given a shape of, for example, (m,n,k), m*n*k samples are generated, and packed in an m-by-n-by-k arrangement. Because each sample is N-dimensional, the output shape is (m,n,k,N). 
print(x.shape) #(3, 3, 2)

# power
print(2**4) # 16

# convert object array to normal array: normal_array = object_array.astype(None)



# numpy.linspace
exlinspace = np.linspace(2.0, 3.0, num=5)  # array([ 2.  ,  2.25,  2.5 ,  2.75,  3.  ])

# numpy.ravel()
x_ravel = np.array([[1, 2, 3], [4, 5, 6]]) 
x_after_ravel = x_ravel.ravel() # Return a contiguous flattened array [1 2 3 4 5 6]

# numpy.add to add 2 numpy arrays elementwise
const = ([4]*56) # list [4,...,4]
const = (np.asarray(const)).reshape(-1,1) # array([4,...,4])

y_plus_const = const+x

# numpy.asarray to convert a list to an array
ex_as_array = [4]*56
ex_as_array = np.asarray(ex_as_array)

# adding a const value elementwise
bbbbb = np.ones(3) + 8 # array([9,9,9])
# limite np.zeros
abbbb = np.array([1,2,3]) # [1,2,3]
resbb = abbbb + bbbbb # [10,11,12]

# *: product elementwise and np.dot: matrix product
astar= np.array([1,1,3])
bstar= np.array([1,2,3])
cstar = astar*bstar # [1,2,9]


# np.repeat
ex_repeat = np.repeat([1,-1], 5) # [1,1,1,1,1,-1,-1,-1,-1,-1]

# np.concatenate((a1, a2, ...), axis=0, out=None : join a sequence of arrays along an existing axis.

a_conc = np.array([[1, 2], [3, 4]])
b_conc = np.array([[5, 6]])
res_conc = np.concatenate((a, b), axis=0)

###
#array([[1, 2],
#       [3, 4],
#       [5, 6]])
   
# numpy.vstack : Stack arrays in sequence Vertically
a_vstack = np.array([1, 2, 3])
b_vstack = np.array([2, 3, 4])
res_vstack = np.vstack((a,b))
###
#array([[1, 2, 3],
#       [2, 3, 4]])	   
	   
# numpy.hstack(tup)[source] : Stack arrays in sequence horizontally (column wise).
a_hstack = np.array((1,2,3))
b_hstack = np.array((2,3,4))
res_hstack = np.hstack(a_hstack,b_hstack) #array([1, 2, 3, 2, 3, 4])




''' List things to know '''

'''
LIST COMPREHENSIONS

[ expression for item in list if conditional ]
*result*  = [*transform*    *iteration*         *filter*]

'''

a_list_comp = [0,2,3,5,6,7,8,9,4,5]

b_list_comp = [i*i for i in a_list_comp if (i%2)==0] # [0,4,36,64,16]
c_list_comp = [i for i in range(10)] # [0,1,2,3,4,5,6,7,8,9]

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
a_gen_object = my_gen()
# We can iterate through the items using next().
next(a_gen_object) #This is printed first
#1
# Once the function yields, the function is paused and the control is transferred to the caller.

# Local variables and theirs states are remembered between successive calls.
next(a_gen_object)#This is printed second
#2

next(a_gen_object)#This is printed at last
#3

# One interesting thing to note in the above example is that, the value of the variable n is remembered between each call.
# Unlike normal functions, the local variables are not destroyed when the function yields. 
# One final thing to note is that we can use generators with for loops directly.

# Using for loop
# for item in my_gen():
#   print(item)  

''' MAP, REDUCE, FILTER, ITER, RANGE, XRANGE '''


''' MAP 

map(function_to_apply, list_of_inputs)

instead of:
    
items = [1, 2, 3, 4, 5]
squared = []
for i in items:
    squared.append(i**2)

just do:
    
items = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x**2, items))
    
'''


# Most of the times we use lambdas with map so I did the same. 
# Instead of a list of inputs we can even have a list of functions!

def multiply(x):
    return (x*x)
def add(x):
    return (x+x)

funcs = [multiply, add]
for i in range(5):
    value = list(map(lambda x: x(i), funcs))
    print(value)

# Output:
# [0, 0]
# [1, 2]
# [4, 4]
# [9, 6]
# [16, 8]

'''

FILTER

filter creates a list of elements for which a function returns true. Here is a short and concise example:

'''

number_list = range(-5, 5)
less_than_zero = list(filter(lambda x: x < 0, number_list))
print(less_than_zero) # Output: [-5, -4, -3, -2, -1]


''' 

REDUCE

Reduce is a really useful function for performing some computation on a list and returning the result. 
It applies a rolling computation to sequential pairs of values in a list. 
For example, if you wanted to compute the product of a list of integers.

So the normal way you might go about doing this task in python is using a basic for loop:

'''
product = 1
list_ex = [1, 2, 3, 4]
for num in list_ex:
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

# time_dec is a decorator
# it takes a function as an argument and generates a new function (augmenting the work of the original function) 
# and it returns this new function
import time

def time_dec(func):
    def wrapper(*arg): #*args in order to accept any arbitrary numberr of arguments and keywords arguments
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



















