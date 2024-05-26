# 第一题

CD

A. 错误, 在不同的scope里面

B. 错误, 因为局部变量声明不会初始化, 因此是未定义状态 

CDE很明显

# 第二题

ABCE  D错误, 因为cp1是指针, 不能用subscript(扔进函数里面倒是可以用), 换而言之, cp1 cp2类型其实是不一样的(但是扔进函数时, 数列退化为指针)

注意E是正确的, `strlen(cp2)` 的值是 5, `sizeof(cp2)` 的值是 `6` 个字节

# 第三题

BC

A错在sizeof(struct) >= sizeof(types......), 因为不是所有类型内存简单相加就是结构体的内存大小 

BC都是正确的, 其中Csize of(struct student)是计算student结构体的大小, 而B中sizeof(*student)的星号其实是解引用指针, 计算指向的`struct student` 类型的变量的大小 

# 第四题

BC

AC: 其实我们希望是对指针做手脚, 那么传入这个指针的时候, 其实是复制了一份临时的指针, 然后指到了对应位置, 但是调用结束的时候就摧毁了. 因此需要传进去指针的指针, 通过指针的指针, 操控在函数体外面的指针的指向. C为什么是对的? 因为返回了最新指向正确位置的指针, 而不是A一样临时的, 正确的指针会随着函数结束而消失

BD: 利用指针的指针来操控指针是正确的, B里面取地址符就是给了一个指针的指针, 而D却是解引用了

# 第五题 

C

A. printf expects the first argument to be a string. Passing NULL to it is undefined behavior

B. If the conversion specifier does not match the argument type, the behavior is undefined.(不允许转换)

D. Overflow happens when calculating ival * ival

# 第六题

AE 细心一点应该就能对; 注意n=3的时候, 会产生n=1的情况(不交换)

# 第七题

quiz涉及过的程序设计漏洞

# 第八题

- `fun(int a[100])` 中的 `a` 是一个退化的数组，你只能通过指针的方式访问数组元素，没有数组的大小信息。
- `fun(int (*a)[100])` 中的 `a` 是一个指针，它保留了数组的大小信息，你可以像使用数组一样使用它，包括访问其大小。

AB输入的``void fun(int a[100])``其实代表fun接受的参数是指针, 因为它等价于``void fun(int *a);``

C代表输入的是数列的引用(意味着函数接受数列作为参数, 而不是指针), 所以使用的时候``fun(a);``

D代表输入的是数列的指针(和输入指针的A有差别!) 因此使用的时候``fun(&a)``

注意D中传入的指针直接可以当作数列使用, 是可以基于范围的循环的那种; C中的a直接就可以当作数组使用

# 第九题

AD 很简单

# 第十题

ACDE

能够基于范围的循环, 必须是容器, array string vector之类的; 指针是不可以的

因此B是不可以的, 说到底a是指针 

E是可以的, 因为传进来的是数组的引用, a是真正的数组, 而F中的a只是退化的指针

# 第十一题

BDF 这道题比较阴间, 但是也反映了很多的知识点 

A. 可以编译. 虽然说const对象代表该实例成员不会被修改, 而且只能调用const方法, 但是实际上方法的声明中最前面的const是可以不加的, 不加也可以编译, 只要最后返回的引用的值不会被修改. 

只不过是在实际的代码中, 毕竟使用const的意义就是防止修改, 所以说方法都是两个const都配套好的. 但是实际上, 方法定义的时候最前面没有加上const也是不会编译报错的, 当然如果修改了这个值的话就会出错

B. 当然

C. 非 `const` 成员也可以调用和使用 `const` 方法; 如果一个方法同时有const和非const, 那么非const实例会优先调用非 `const` 版本的成员函数。所以说可以编译

DE . `operator[]` 这样的函数被设计为返回非常量引用(返回的是int&, not const int&), 允许对对象的某些部分进行修改。在这种情况下，虽然对象本身是 const，但是通过这些函数可以修改对象的某些成员。

F. C++ 中不能返回数组的引用; 而operator[] 将返回一个 int&，这是一个数组元素的引用，而不是指针

# 第十二题

AD   比较阳间

AD: 全局变量会零初始化, 而new int[10]{}的{}代表零初始化

E: 其实是1,0,0,...; int a[10]{1} 也是初始化第一个元素为1，其余元素会被默认初始化为0

# 第十三题 

AB

A. 如果用户定义了一个构造函数, 那么编译器不会再提供默认构造函数

B. 如果initializer里面没有初始化一个成员, 那么就会看这个成员在in-class有没有默认初始值 

C. 初始化的顺序必须是声明的顺序

D. 如果成员是const,,  那么这个成员只能在初始化列表里面初始化 

# 第十四题

AB

D: std::string  std::vector不需要手动释放, 因为STL会自动释放

C: The copy constructor, the move constructor and the destructor of Book are implicitly declared, and implicitly defined if they are used. The copy assignment operator and the move assignment operator are implicitly declared and implicitly deleted since title and isbn are const. The word ”has” in choice C is not clear.

在 C++ 中，如果没有显式地声明或定义某些特殊成员函数（如拷贝构造函数、移动构造函数和析构函数），编译器会自动生成这些函数。这些函数负责对象的复制、移动和销毁。因此，即使没有明确定义，编译器也会为 `Book` 类隐式声明和定义这些函数，以便在需要时执行相应的操作。因此，即使没有明确提及这些函数，它们仍然存在并起作用。

因为 `title` 和 `isbn` 是 `const` 类型的成员变量，它们一旦初始化就不能被修改。在 C++ 中，如果一个类拥有常量成员变量，那么相应的拷贝赋值运算符和移动赋值运算符会被隐式声明为已删除的函数。这是因为默认的赋值操作无法修改这些成员变量，因此将它们声明为已删除的函数可以避免错误的赋值行为。

理解为什么"赋值运算"不可以, 因为这是属于"修改类型"; 但是赋值构造却可以, 因为说到底是在"构造". 构造和修改是两件事情(就像为什么const成员不能在函数体里面定义, 必须在初始化列表里面初始化)

# 第十五题

ACD

D. 	`emplace_back` 接受的参数是用于构造新元素的参数列表。它将这些参数传递给元素类型的构造函数，然后在向量的末尾直接构造新元素。接受参数之后, 因为vector里面全是Book, 所以这些参数用于Book实例的初始化了

# 第十六题

C

D错误, 因为Item的析构函数是默认析构函数, 并不是虚函数; 而只有基类和派生类的析构函数为虚函数的情况下, 在静态类型是基类但是动态类型是派生类的指针的释放过程中, 指针释放才会调用派生类的析构函数 

所以说如果Item定义了析构函数, 且派生类里面override了, 那么这个选项就是正确的

# 第十七题 

AD   

B是错误的: 因为其实snacks里面都是Item的智能指针; 调用的时候, 还是会受限于Item的范围内; 除非item里面定义了set_discount()函数, 且是虚函数, 然后派生类里面override; 那么这种情况下, Item为静态类型的指针可以通过虚函数关系访问到动态类型的DiscountItem的set_discount()方法.

总而言之, 这一类题目的关键: 注意智能指针静态类型是不是基类; 如果是, 那么如果想访问作为动态类型的派生类里面的方法, 这个方法必须是基类里面定义为虚函数, 然后派生类里面覆写

# 第十八题 

B   基类声明了是虚函数, 那么派生类里面什么关键词都不加都可以

但是实战中最好还是virtual override关键词都加上, 增加可读性 

而且: 传参列表要一样, 返回值类型要一样, 而且const-ness要一致!

# 第十九题 

BC

A. unary operation在类里面定义的话, 什么参数都没有; 但是如果是在类外定义的话, 参数有一个

````c++
#include <iostream>
// 类里面定义 
class MyClass {
private:
    int value;
public:
    MyClass(int val) : value(val) {}
    MyClass operator-() const {
        return MyClass(-value);
    }
};
// 或者用友元
class MyClass {
private:
    int value;
public:
    MyClass(int val) : value(val) {}
    friend MyClass operator-(const MyClass& obj);
};
// 友元函数定义
MyClass operator-(const MyClass& obj) {
    return MyClass(-obj.value);
}
````

BC. 因为二元操作符在类外定义, 而且没有explicit, 而且操作是symmetric的, 所以说正确

D. 友元不属于类里面的成员 

# 第二十题

B   注意返回的应该是对象实例, 而不是this指针, 所以说返回的*this就是在对this指针解引用, 返回对象本身以实现链式编程

# 第二道大题:

````c++
auto checkPeak(const std::vector<int> &v) {
  std::size_t i = 0;
  while (i + 1 < v.size() && v[i] <= v[i + 1])
    ++i;
  auto j = v.size() - 1;
  while (j > 0 && v[j] <= v[j - 1])
    --j;
  return i > 0 && i + 1 < v.size() && i == j;
}
auto negate(std::vector<int> v) {
  for (auto &x : v)
    x = -x;
  return v;
}
bool isSinglePeaked(const std::vector<int> &v) {
  return checkPeak(v) || checkPeak(negate(v));
}
````

有点算法的味道; 第一个函数是判断是不是"山峰"类型, 两个iterator从左边和右边分别出发

第二个函数是反转, 因为可能符合要求的是"谷底", 谷底反过来就是山峰

最后判断是不是山峰或谷底

# 第三道大题:

(1) ``virtual ~Clock() = defualt``

(2)  (1)处:

 ````c++
 friend std::ostream &operator<<(std::ostream &, const Time&);
 Time &operator++(){
     ++m_minute;
     if (m_minute == 60) {
         m_hour = (m_hour + 1) % 24;
 		m_minute = 0;
     }
     return *this;
 }
 ````

​       (3)处: 

````c++
std::ostream &operator<<(std::ostream &, const Time &t){
    return os << Time::fill2(t.m_hour) << ":" << Time::fill2(t.m_minute);
}
````

(3) 

````c++
// (1) 加上判断两个Time是否相等的运算符重载
bool &operator==(const Time &rhs) const{
    return m_hour == rhs.m_hour && m_minute == rhs.m_minute;
}
// (2) display
virtual void display() const override{
    Clock::display(); // 基类的方法是继承的
	std::cout << "Alarm: " << m_alarm << std::endl;
	if (m_alarm == m_time)
		std::cout << "ALARM!" << std::endl;
}
````

# 第四道大题

````c++
Dynarray sorted() const & {
	auto ret = *this;
	std::sort(ret.m_storage, ret.m_storage + ret.m_length);
	return ret;
}
Dynarray &&sorted() && {
	std::sort(m_storage, m_storage + m_length);
	return std::move(*this);
}

````







