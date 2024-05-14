# 装饰器

## 引子

见下面案例:

````python
def func():
    print('我是func函数')
    value = (11,22,33,44)
    return value
result = func()
print(result)
````

假设我有一个需求, 我需要函数在执行之前, 输入before, 执行后输出after

一般来说正常思维是这样

````python
def func():
    print('before')
    print('我是func函数')
    value = (11,22,33,44)
    print('after')
    return value
result = func()
print(result)
````

但是有更好的方法:

````python
def func():
    print('我是func函数')
    value = (11,22,33,44)
    return value
def outer():
    def inner():
        pass
    return inner
func = outer()
result = func()
print(result)
````

func在下面变成了outer里面定义的inner, 不再是原来的func

我们再进行一些修改:

````python
def func():
    print('我是func函数')
    value = (11,22,33,44)
    return value
def outer(origin):
    def inner():
        res = origin()
        return res
    return inner
func = outer(func)
result = func()
print(result)
````

给outer函数传入了一个参数, 而func = outer(func) 是把func当做参数传入了outer函数

这样就形成了一个完整的闭包:  func - outer - inner - func

func变成outer, 传入的参数是自己, 然后自己变成了outer定义的inner函数, 在上面代码中, 执行这个函数相当于执行了原来的func函数

这样套娃究竟是为了什么? 这是为了更好地对这个函数进行修改:

````python
def func():
    print('我是func函数')
    value = (11,22,33,44)
    return value
def outer(origin):
    def inner():
        print('before')
        res = origin()
        print('after')
        return res
    return inner
func = outer(func)
result = func()
print(result)
````

func变相执行的inner函数中, 实现了功能的同时, 原来的代码被正常地执行了

## 真正的装饰器

上面的代码虽然有闭包, 虽然功能实现了, 但是非常的长, 显得很没有优势

但是其实Python内部由专门的简化装饰器的语法:

python中支持特殊语法: @函数名

````python
'''
@函数名():
def xxx():
	pass
python内部会自动执行 函数名()
并且将下面定义的xxx函数自动当做参数传入进去, 并且执行完之后将结果赋值给这个xxx函数
相当于执行xxx = 函数名(xxx)
那么上面这个代码优化之后的代码是什么呢? 
'''
def outer(origin):
    def inner():
        print('before')
        res = origin()
        print('after')
        return res
    return inner
@outer
def func():
    print('我是func函数')
    value = (11,22,33,44)
    return value
result = func()
print(result)
````

看似代码还是不是非常简洁, 但是我们考虑不一样的要求:

我们有三个函数, 各自都在输出结果前和后加上before and after

这样的话, 如果用简单的方案

````python
def outer(origin):
    def inner():
        print('before')
        res = origin()
        print('after')
        return res
    return inner
@outer
def func1():
def func2():
def func3():
# 这样就批量处理了多个函数
````

## 优化支持n个参数

在outer中, 传入的参数不一样; 在inner函数中, 找origin找不到, 那么就会找到传入的作为参数的函数

那么其实还有可以优化的地方:

如果我们参入的函数有参数? 那么应该怎么办? 

````python
def outer(origin):
    def inner():
        print('before')
        res = origin()
        print('after')
        return res
    return inner
@outer
def func1(a1):
    value = a1
    return value
def func2(a1,a2):
    value = a1+a2
    return value
'''
这样的代码一定会报错, 因为inner接受的参数的数量并不清楚
所以要想办法让inner能够接受的各种各样的参数
而却原先会将执行的origin也必须能够接受所有的各式各样的参数
'''
def outer(origin):
    def inner(*args,**kwargs):
        print('before')
        res = origin(*args,**kwargs)  # 注意后面加括号是调用了
        print('after')
        return res
    return inner
@outer
def func1(a1):
    value = a1
    return value
def func2(a1,a2):
    value = a1+a2
    return value
````

## 总结

实现原理: 基于@语法和函数闭包, 将原函数封装在闭包里面, 然后将函数赋值为一个新的函数(内层函数), 执行函数时再在内层函数中执行闭包的原函数

实现效果: 可以在不改变原函数的内部和调用方式的前提下, 实现在函数执行和执行扩展功能

使用场景: 多个函数统一在执行前后自定义一些功能和内容



