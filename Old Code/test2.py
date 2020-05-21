from test1 import imported_func1, func1
import test1

def func2():
    print("I am func2 trying to override func1.")

test1.func1 = func2

imported_func1()
