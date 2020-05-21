# i define a function here which references a function here

def func1():
    print("I am the original func 1.")

def imported_func1():
    print("I am the importd func1 printing func1")
    func1()
