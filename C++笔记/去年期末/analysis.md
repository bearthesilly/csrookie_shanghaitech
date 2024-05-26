# 第一题

答案是AD; BC一看就是错的

A: **宏定义** (`#define`): 允许定义宏，它可以是常量值、表达式或代码块。在编译之前，这些宏会被替换为它们的定义。这是Preprocessor能干的事情. 

````c
const int N = 100; // 常量表达式
int a[N], b[N]; // 编译器在编译时可以确定数组大小
````

同时, 上面这种也是正确的, 因为N也是常量; 但是如果是``int N = 100``就会报错

D: 在 C 语言中，局部数组如果不显式初始化，默认会初始化为`0`。因此，`int a[N], b[N];`中的`a`和`b`的元素会被初始化为`0`。

但是如果是全局数组或者是静态数组(定义在函数外部或者是static定义的数组), 那么它们不会自动初始化, 他们的初始值是未定义的 

除此之外, malloc开辟内存的时候不会零初始化, 但是calloc会

# 第二题

AD   B是错的, 因为sizeof(str)返回的是指针的大小, str本质上还是一个指针 

C也明显是错的, 因为不是回文的

A是对的, 确实不修改字符串, 那么传入的时候可以加上const

# 第三题

E

A. 这里只是形式传参, 根本不会修改原来的结构体

B. 这里只是将指针的指向改变了, 原始指针所指向的内存区域并没有被修改

C. 在函数体里面创建了一个结构体, 但是返回值之后, 这个结构体就会被清除掉, 因为虽然是在堆区开辟的data, 但是result指针是开辟在栈区, 函数结束的时候就会被清除掉 

D. 语句 `free(v)` 不会自动释放为 `v.data` 分配的内存。要释放为 `v.data` 分配的内存，需要显式调用 `free(v.data)`

想要C是正确的, 可以尝试如下修改: 

````c
struct Vector *vector_add(const struct Vector *lhs, const struct Vector *rhs) {
    size_t dim = lhs->dim < rhs->dim ? rhs->dim : lhs->dim;
    struct Vector *result = malloc(sizeof(struct Vector)); // Allocate memory for the result vector
    if (result == NULL) {
        // Handle memory allocation failure
        return NULL;
    }
    result->dim = dim;
    result->data = calloc(dim, sizeof(double)); // Allocate memory for the data array
    if (result->data == NULL) {
        // Handle memory allocation failure
        free(result); // Free the allocated memory for the result vector
        return NULL;
    }
    for (size_t i = 0; i < lhs->dim; ++i) {
        result->data[i] += lhs->data[i];
    }
    for (size_t i = 0; i < rhs->dim; ++i) {
        result->data[i] += rhs->data[i];
    }
    return result; // Return the pointer to the dynamically allocated result vector
}
````

上面代码中, result结构体指针就是开辟在堆区, 留意是如何在堆区开辟结构体的: 

``struct Vector *result = malloc(sizeof(struct Vector));``

# 第四题

B

A: ``1ll``代表long long数据类型的1, 那么右边的式子其实数据类型将会是Long long, 而不是``int``

B: 这个语法将会导致undefined behavior: 不能对一个变量同时使用+修改!

C: `printf` 函数的格式化输出要求参数的类型必须与格式化字符串中的格式说明符相匹配。当您使用 `%d` 格式说明符时，它期望的参数类型是 `int`。然而，`float` 类型与 `int` 类型是不兼容的，因此不能隐式转换. 总而言之, ``printf``这个函数就是强制要求类型匹配, 不容忍隐式转换 

D: 程序会直接崩溃

# 第五题

C, 没什么好说的

# 第六题

ACD

B错是因为初始化方式错了, 应该是: (n, m为int)

``std::vector<std::vector<double>> matrix(n, std::vector<double>(m));``

# 第七题

ACD  哪些能够使用基于范围的循环呢? **标准库容器**, 以及传统数组

但是``int *a = new int[100]{};``说到底, 还是只是一个指针, 而``int a[100]{};``中a的数据结构还是数组, 只不过是传入函数的时候会退化为指针 

# 第八题 

ABD  

A: 与结构体不同, class成员默认是private的

B: const代表``const Dynarray``实例也能调用这个函数, 而``const Dynarray``代表这个实例的值是不能被修改的, 因此这个指针其实是``int *const``, 代表指向的值不能被修改. 而说实话, 在const实例中, this指针的类型是``const Dynarray *`` . 

 D: 保证可能的``const Dynarray``实例的需求

# 第九题

A: 有默认构造函数

B: 初始化顺序必须是正确的, 不然会报错

C: **如果类中没有用户定义的构造函数或析构函数**，编译器将自动生成一个默认构造函数（如果类中没有其他构造函数）。这个默认构造函数将执行默认初始化，对于类中的非静态成员变量，这通常意味着：

- 对于内置类型（如 `int`、`double`、`bool` 等），成员变量将被零初始化（即设置为0、0.0、`false` 等）。
- 对于类类型（即用户定义的类型），成员变量将进行默认初始化，这通常意味着它们自己的默认构造函数将被调用。

D: 很正确

# 第十题 

AB

A: 非常有道理, 因为是不会修改值的, 所以说``const Book``实例有这个函数需求

B: 完全正确, 注意运算符的调用是如何等价过去的

C: 类成员函数仍然有权访问类的私有成员

D: 类外定义, 要两个实例都要传进来

# 第十一题

C

A: 错误! 因为create函数是一个静态函数, 它没有this指针!

B: 错误! 应该是``std::unique_ptr<Book>``

C: 正确! 构造函数都是私有成员, 所以说只能用公开的create函数构造

D: 错误!  `std::make_unique` 模板不接受任何形式的移动语义操作符 `std::move` 作为其参数

# 第十二题

BC

A: 错误, 虽然函数(2)只接受一个参数，但它被重载为一个二元运算符，用于实现两个 `Complex` 对象的减法。在C++中，一元负号运算符（-x）是一个非成员函数，并且其参数是 `Complex` 类型的引用。如果(2)是一元负号运算符，它应该被声明为 `Complex operator-() const;`

B: 允许隐式使用, 而0隐式转换之后能够用"+", 但是乘法没有

C: 明显正确 

D: 函数(3)是 `operator+` 的重载，它是 `Complex` 类的一个友元函数，而不是成员函数。友元函数不属于类，它们只是被赋予了访问类的私有和保护成员的能力

# 第十三题

BD

A: 没有正确的初始化result

B: 当 `std::copy_if` 函数迭代输入范围并决定是否复制某个元素时，它会调用传递给它的谓词函数。在这种情况下，谓词是 `LessThanK` 对象的 `operator()` 函数. 在 `std::copy_if` 的每次迭代中，都会调用 `LessThanK` 对象的 `operator()` 函数来检查当前元素是否满足条件（即是否小于 `k`）。如果满足条件，元素就会被复制到 `result` 向量中。

C: 完全错误, 传参都没有, 仅仅是放了一个函数在那里

D: ``[k](int x) { return x < k; }``

1. **捕获子句 `[k]`**：这个部分定义了 lambda 表达式如何捕获其外部作用域中的变量。在这里，它使用一个捕获列表来捕获变量 `k`。这意味着 lambda 表达体可以访问并且使用变量 `k` 的值。
2. **参数列表 `(int x)`**：这个部分定义了 lambda 表达式接受的参数。在这个例子中，它接受一个类型为 `int` 的参数，命名为 `x`。
3. **函数体 `{ return x < k; }`**：这是 lambda 表达式的函数体，它定义了当 lambda 被调用时应该执行的操作。在这个例子中，函数体只包含一个返回语句，它返回一个布尔值，这个布尔值是 `x < k` 表达式的结果。这里，`x` 是传递给 lambda 的参数，而 `k` 是从外部作用域捕获的变量。

# 第十四题 

CD

A. 所有的东西都被继承了, 但是访问权限会有所差别

B. 初始化列表第一项必须是调用基类的构造函数 

C. 完全正确

D. 可以, ``std::move``没什么问题 

# 第十五题

D. virtual在基类里面是强制的, 在派生类里面是optianal; 但是派生类里面一定要override覆写

 
