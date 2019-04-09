''' Underscore in Python '''

''' Example '''
class A:
    def __init__(self,x):
        self.x = x
    def imprime(self):
        print("Normal : x = "+str(self.x))
    def _imprime(self):
        print("Single underscore : x = "+str(self.x))

a = A(10)
a.imprime() # Normal : x = 10
a._imprime() # Single underscore : x = 10

'''
._variable is semiprivate and meant just for convention

.__variable is often incorrectly considered superprivate, while it's actual meaning is just to namemangle to prevent accidental access

.__variable__ is typically reserved for builtin methods or variables
(The Python interpreter has a number of functions that are always available for use. 
These functions are called built-in functions. 
For example, print() function prints the given object to the standard output device (screen) or to the text stream file))

You can still access .__mangled variables if you desperately want to. 
The double underscores just namemangles, or renames, the variable to something like instance._className__mangled

Example:
'''

class Test(object):
    def __init__(self):
        self.__a = 'a'
        self._b = 'b'

t = Test()
print(t._b) #'b' .t._b is accessible because it is only hidden by convention

'''t.__a

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
AttributeError: 'Test' object has no attribute '__a'

t.__a isn't found BECAUSE IT NO LONGER EXISTS DUE TO NAMEMANGLING '''

print(t._Test__a) #'a'. By accessing instance._className__variable instead of just the double underscore name, you can access the hidden value

print(vars(t)) #{'_Test__a': 'a', '_b': 'b'}

#By accessing instance._className__variable instead of just the double underscore name, you can access the hidden value

''' _ALL__ 

is a list of public objects of that module, as interpreted by import *. 
It overrides the default (which is import *) of hiding everything that begins with an underscore.

In summary:
    
    
- It tells the readers of the source code — be it humans or automated tools — what’s the conventional public API exposed by the module.

- It lists names to import when performing the so-called "wild import": from module import *.

'''



''' LAMBDA FUNCTIONS 

DEFINITION

A lambda function is a small anonymous function, a function that is defined without a name.
While normal functions are defined using the def keyword, in Python anonymous functions are defined using the lambda keyword.

SYNTAX

Syntax of Lambda Function in python

lambda arguments: expression
A lambda function can take any number of arguments, but can only have one expression.
The expression is evaluated and returned

EXAMPLE

An example of lambda function that doubles the input value.
'''

double = lambda x: x * 2

print(double(5)) # Output: 10

def say_hello():
    return "hello"
     
# OR
     
say_hello = lambda:"hello"

print(say_hello()) # hello

''' Lambda functions can take any number of arguments: '''
x = lambda a, b : a * b
print(x(5, 6))

'''
Why Use Lambda Functions?

The power of lambda is better shown when you use them as an anonymous function inside another function.

Say you have a function definition that takes one argument, and that argument will be multiplied with an unknown number:

'''
def myfunc(n):
  return lambda a : a * n


''' Use that function definition to make a function that always doubles the number you send in: '''


def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)

print(mydoubler(11))

''' Or, use the same function definition to make a function that always triples the number you send in: '''

def myfunc(n):
  return lambda a : a * n

mytripler = myfunc(3)

print(mytripler(11))

''' Or, use the same function definition to make both functions, in the same program: '''

def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)
mytripler = myfunc(3)

print(mydoubler(11))
print(mytripler(11))

''' In summary: use lambda functions when an anonymous function is required for a short period of time.'''



''' TERNARY OPERATOR '''
a = 5; b = 4
answer = "a is greater than b" if a > b else "a is less than b"
print(answer) # a is greater than b


''' OTHER '''
my_liste = filter(lambda x: str(x)[0] in ('1','3'), range(0,99))

example_lamda = lambda x: x*2

print(example_lamda(2))



''' What does this stuff mean: *args, **kwargs? And why would we use it? 

Use *args when we aren't sure how many arguments are going to be passed to a function,
or if we want to pass a stored list or tuple of arguments to a function. 

**kwargs is used when we dont know how many keyword arguments will be passed to a function,
or it can be used to pass the values of a dictionary as keyword arguments. 

The identifiers args and kwargs are a convention, you could also use *bob and **billy but that would not be wise.

Here is a little illustration:
    
-arbitrary number of POSITIONAL ARGUMENTS: *args
(position matters)
-arbitrary number of KEYWORD ARGUMENTS: **kargs
(position does not matters)

More clear and obvious to use keyword arguments

Why use *args and **kargs:
    
- to pass an arbitrary number of arguments (positional or keywords) as mentioned above
- time saver
'''


def f(*args,**kwargs):
    print(args, kwargs)


l = [1,2,3]
t = (4,5,6)
d = {'a':7,'b':8,'c':9}

f()
f(1,2,3)                    # (1, 2, 3) {}
f(1,2,3,"groovy")           # (1, 2, 3, 'groovy') {}
f(a=1,b=2,c=3)              # () {'a': 1, 'c': 3, 'b': 2}
f(a=1,b=2,c=3,zzz="hi")     # () {'a': 1, 'c': 3, 'b': 2, 'zzz': 'hi'}
f(1,2,3,a=1,b=2,c=3)        # (1, 2, 3) {'a': 1, 'c': 3, 'b': 2}

print(help(sorted))
print(sorted([1,5,8,5,3,1],reverse = False)) # If just "False" then TypeError: must use keyword argument for key function


def ex_key(*args):
    res= 1
    for i in args:
        res = res*i
    print(res)

print(ex_key(1,2,3)) # 6

