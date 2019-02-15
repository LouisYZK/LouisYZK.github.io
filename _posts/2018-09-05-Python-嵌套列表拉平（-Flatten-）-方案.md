---
layout:     post
title:      Python嵌套列表拉平（Flatten）方案
subtitle:   Be pythonic, be simple.
date:       2018-09-05
author:     Louis Younng
header-img: img/post-bg-rwd.jpg
catalog: true
tags:
    - Python
---
## Why flatten?
方便遍历/map/reduce操作，其实就是降维。
## 层级列表因何产生？
写爬虫多的朋友经常会用到批处理的方式做大量的数据请求，返回的批次数据就保存在了一个list，而全部数据通常采用写入文件/数据库，或存在内存的方式；后者就不可避免地做大量append操作。其次对接口结果多样，甚至可能达到多层嵌套的情况；
## 二级嵌套
二级嵌套是最常见的。处理办法也有很多：
```python
import functools
import itertools
import numpy
import operator
import perfplot


def forfor(a):
    return [item for sublist in a for item in sublist]


def sum_brackets(a):
    return sum(a, [])


def functools_reduce(a):
    return functools.reduce(operator.concat, a)


def itertools_chain(a):
    return list(itertools.chain.from_iterable(a))


def numpy_flat(a):
    return list(numpy.array(a).flat)


def numpy_concatenate(a):
    return list(numpy.concatenate(a))


perfplot.show(
    setup=lambda n: [list(range(10))] * n,
    kernels=[
        forfor, sum_brackets, functools_reduce, itertools_chain, numpy_flat,
        numpy_concatenate
        ],
    n_range=[2**k for k in range(16)],
    logx=True,
    logy=True,
    xlabel='num lists'
    )
```
最后的效率对比：
![](https://i.stack.imgur.com/Bz30A.png)
可以看到各种方式在数据量级别上的效率均有差异。用到的工具也有很多。
## 普适的Pythonic写法
对于多级层级列表，普适的写法为;
```python
def flatten(items):
    for x in items:
        # 终止条件，检验是否为可迭代对象
        if hasattr(x,'__iter__') and not isinstance(x, (str, bytes)):
            #Python2写法
            # for sub_x in flatten(x):
            # yield sub_x
            #Python3写法
            yield from flatten(x)
        else:
            yield x
le = list(flatten(list))
```
以上是一个生成器函数，可以简单地理解为用递归的思想遍历了层级列表

关于此，python的**生成器（generator）、迭代器（Iterator）、可迭代对象（Iterable）**的概念就要再复习一遍
简单关系:
![](https://pic4.zhimg.com/80/v2-95b4076d30e55da078045cdade28cea3_hd.jpg)

但是这张图还是没能说明白迭代器和可迭代对象上的功能区别：
我觉得廖雪峰的教程这段话很好：
> Python的Iterator对象表示的是一个数据流，Iterator对象可以被next()函数调用并不断返回下一个数据，直到没有数据时抛出StopIteration错误。可以把这个数据流看做是一个有序序列，但我们却不能提前知道序列的长度，只能不断通过next()函数实现按需计算下一个数据，所以Iterator的计算是惰性的，只有在需要返回下一个数据时它才会计算。

这就是批处理时经常用生成器的原因

Iterator甚至可以表示一个无限大的数据流，例如全体自然数。而使用list是永远不可能存储全体自然数的。

参考 [StackOverflow关于二层嵌套列表拉平的python代码讨论](https://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python#)