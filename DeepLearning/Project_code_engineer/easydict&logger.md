# Easydict

`easydict` 是一个 Python 库，它允许你将字典的键作为属性来访问。这在深度学习和其他编程领域中非常有用，因为它提供了一种方便的方式来管理配置参数。

例如，如果你有一个配置字典，使用 `easydict` 后，你可以这样访问字典中的值：

````python
from easydict import EasyDict as edict

config = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'num_epochs': 10
}

config = edict(config)

# 现在可以像访问属性一样访问字典的值
print(config.learning_rate)  # 输出: 0.001
print(config.batch_size)     # 输出: 64
print(config.num_epochs)     # 输出: 10
````

使用 `easydict` 可以使得代码更加清晰和易于维护，特别是在处理复杂的配置参数时。在深度学习框架中，配置参数通常很多，使用 `easydict` 可以简化参数的访问和管理。

如果没有这个库, 那么如何访问字典的值? 

````python
print(config['learning_rate'])  # 输出: 0.001
print(config['batch_size'])     # 输出: 64
print(config['num_epochs'])     # 输出: 10
````

# logger

