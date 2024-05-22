1. C语言中的每个变量都有一个类型，这个类型在编译时就已确定（静态类型）

2. 获得'9'的int数值: ``int num = '9' - '0';``

3. 有符号整数溢出是未定义行为。无符号整数运算总是以模 `n` 进行，`n` 是整数类型的位数。

4. Whether char is signed or unsigned is implementation-defined.

5. printf函数要求数据类型与占位符严格对应, 不允许隐式转换

6. ``printf("%d\n", ++x);``可以, 但是``printf("%d%d\n", x, x++);``不行, 因为**不知道**是先占位还是先++x

7. Operator precedence does not determine evaluation order.  ``f() + g() * h()`` is interpreted as ``f() + (g() * h())`` , but the order in which ``f , g and h ``are called is ***unspecified***.

8. In the following expressions, it is ***unspecified*** whether f is called before g .
   ``f() + g() ; f() == g() ; some_function(f(), g())``

9. The assignment operator returns the value of lhs after assignment. 

   ````c
   int a = 0, b = 1, c = 2;
   a = b = c; // interpreted as a = (b = c)
   // Both a and b are assigned with 2.
   ````

10. The expression in a **case** label must be an ***integer constant expression, whose value is known at compile-time***

11. If a variable is declared without explicit initialization: 

    For local non- static variables, they are initialized to indeterminate values. In other words, they are uninitialized. (局部非静态变量, 不会零初始化)

    For global or local static variables, they are empty-initialized : (全局或局部静态变量会零初始化)

    Pointers are initialized to null pointer values of their types. 

    Objects of integral types are initialized to 0 .

    Objects of floating types are initialized to positive zero ( 0.0 ). 

12. 局部非静态指针是未初始化的, 而全局或局部静态指针会零初始化(空指针)

13. 要避免解引用空指针或者是野指针, 最好是``if (ptr != NULL && *ptr == 42){}``

14. ````c
    int a1[10]; // OK. A literal is a constant expression.
    #define MAXN 10
    int a2[MAXN]; // OK. `MAXN` is replaced with `10` by the preprocessor.
    int n; scanf("%d", &n);
    int a[n]; // A C99 VLA (Variable-Length Array), whose length is
    // determined at runtime. 不可以!
    ````

15. If an array is declared without explicit initialization: 

    Global or local static : ***Empty-initialization*** Every element is empty-initialized. 

    Local non- static : Every element is initialized to ***indeterminate values*** ***(uninitialized).***

16. 总而言之, 局部变量不初始化, 就是真的未初始化了; 其他的都是空初始化

17. an array can be implicitly converted to a pointer to the first element: ``a -> &a[0] , T -> [N] T * ``; 但是两者仍然有区别, array支持基于范围的循环, 但是如果是函数传参, 传入了一个退化为指针的数组, 那么这个指针不支持基于范围的循环. 只不过是在函数里面的这个指针支持subscript, 因为编译器帮了大忙

18. ``int (*parr)[N];``a ptr to an array of N ints   

    ``int *arrp[N];`` an array of N ptrs pointing to int.

19. 传入`` int[N][M]``的方法:  ``void fun(int a[N][M]) ``  ``void fun(int (*a)[M])`` 一定要理解到底传入的关键是什么? ***A pointer to int [M]!***

20. ``void *malloc(size_t size)``, 不会空初始化, 而calloc会. 有且只有一种free方法: free指针

21. C风格字符串: It must be null-terminated: There should be a null character '\0' at the end. ``'\0'`` is the "null character" whose ASCII value is 0. ``fgets(str, 100, stdin);  ``    puts(str) : Prints the string str , followed by a newline.

22. 想开辟一个结构体实例在堆区上: ``struct Student *pStu = malloc(sizeof(struct Student));``

23.  Although an array cannot be copied, **an array member can be copied**. The copy of an array is **element-wise copy.** (struct)

24. Global or local static : ***"empty-initialization", which performs member-wise empty-initialization.*** (空初始化)

​		Local non- static : every member is initialized to indeterminate values (in other 		words, uninitialized). (未初始化)

25. Default-initialization of a std::string will produce an empty string. (不是undefined)
26. ``string.size()``返回字符串长度; ``string.empty()``判断是否是空字符串
27. At least one operand of ``+`` should be ``std::string ``
28. References must be bound to existing objects; References are not objects
29. Use references in range- for: ``for (char &c : str){for (char &c : str)}``
30. ``std::vector`` is not a type itself. It must be combined with some  to form a type.
31. C++中, ``const int maxn = 1000; int a[maxn]; // a normal array in C++, but VLA in C``

32. Almost all implicit narrowing conversions in C++ is banned. 

33. ````c++
    const int cival = 42;
    int &ref = const_cast<int &>(cival); // compiles, but dangerous
    ++ref; // undefined behavior (may crash)
    ````

34. ``auto str = "hello"; // `const char *`` ``auto it = vs.begin();``

    ``auto lam = [](int x, int y) { return x + y; } // A lambda expression.``

35. decltype(expr) will deduce the type of the expression expr without evaluating it.

36. Pass an array by reference: 

````c++
void print(const int (&arr)[100]) {
	for (auto x : arr) // OK. `arr` is an array.
		std::cout << x << ' ';
	std::cout << '\n';
}
````

37. In a const member function, calling a non- const member function on *this is not allowed.
38. For a const object, only const member functions can be called on it.
39. Data members are initialized **in order in which they are declared**.
40. 类中声明的指针是***无法***被默认初始化为 nullptr(惨痛教训)(=default)(除非in-class有定义)
41. ``new [0] ``may also allocate some memory which should also be deallocated
42. If the class does not have a user-declared copy constructor, the compiler will try to synthesize one. The synthesized one will copy-initialize all the members
43. By saying ``= delete`` , we define a deleted copy constructor. (拒绝复制构造)
44. ``a = b`` is equivalent to ``a.operator=(b) ``.
45. ``operator=`` returns reference to the **left-hand** side object. It is ``*this``.别忘了delete
46. copy和move问题都需要考虑self-assignment的corner-case
47. A ``static`` data member: 它属于类, 而不属于任何object, 因此静态类方法没有this指针
48. A ``friend`` is not a member! 友元声明在类的任何地方都可以(一般是最开始或最后)
49. A member function can be declared in the class body, and then defined outside.
50. 只声明一个类, 但是啥也不知道, 那么就是incomplete type, 只能给它声明指针或引用
51.  析构函数结束后, 类内所有成员都会被自动摧毁, 摧毁顺序是声明顺序的逆序!
52. 默认析构函数**不会**自动释放指针指向的内存
53. 



















