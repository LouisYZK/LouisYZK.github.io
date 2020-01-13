---
layout:     post
date:       2020-01-13
tag:        note
author:     BY Zhi-kai Yang
---
## 重新思考程序级异步--以常见异步场景为例比较asyncio和goroutine方案

> **Preface**
>
> 2019年差不多也是这个时候，我写了[一篇文章]([https://louisyzk.github.io/notes/2019/02/15/Python%E5%BC%82%E6%AD%A5IO%E6%8A%80%E6%9C%AF%E6%A6%82%E8%A7%88](https://louisyzk.github.io/notes/2019/02/15/Python异步IO技术概览))概述python语言本身的异步机制。随着认知的增长，重读文章和当时为了说明问题的实验代码，我发现了很多理解不到位甚至是错误的地方。故写此文一方面来勘误，另一方面也扩展一些东西。去年(2019年) 我接触了**Go语言 (Go Lang)**，同时在实习的项目中多次使用了python多线程实现异步，对于异步常见场景的使用和解决方案都有了一些直观的认识。
>
> 故作此文，比较常见的异步案例来讲述**程序语言**级别的异步实现。主要比较python的`asyncio`方案和go语言的`goroutine`。两者实现的原理不同、使用范式也不太一样。但共同点是他们是从编程语言层面实现了异步，i.e. 没有使操作系统来完成子程序的上下文切换和线程切换。本文不着重探究两种语言方案的实现原理，主要探索如何在对应场景下使用。相信读完本文后读者大致会对`asyncio`和`goroutine`的编程风格有一大致了解。

## 程序级别异步概述

首先还是要简述下什么是异步，以及异步重要的原因。

异步是解决**阻塞(Block)**程序的一种方式，因此先介绍下什么样的程序是阻塞的。

阻塞程序一般会造成CPU空闲，程序停止等待IO操作。常见的IO操作有：数据库连接、数据库增删改查、文件的打开与读取、http连接与请求(底层表现为socket连接与数据接收)；常见的场景如爬虫中Web数据流的接收(相信这个场景是大多数人尝试写过并且熟悉的)，你有很多页面(任务)需要爬取，而http只能建立对每个页面建立一次连接然后传输数据。阻塞的程序会线性地执行每一个页面的"数据接收--处理"操作，而数据接收阶段是不占用CPU核心的，这就导致了算力的浪费。

非阻塞程序不会因为某一操作能停止执行(CPU空闲)，而会保证CPU核心始终在工作状态且符合我们的逻辑需求。上面爬虫的例子中，如果在等待某一页面数据接收的同时，程序能执行其他页面的处理操作，这就是非阻塞程序。我们把使得程序非阻塞的方式称为**异步(asynchrounous)**.

实现异步最简单直接的方式即通过多线程、多进程。即将能够阻塞程序的耗时的IO操作挂起到一个subprocess\subthread, 而主线程\进程执行主要的逻辑。这就需要操作系统来完成线程的切换和子例程上下文的切换。

而程序语言级别实现的异步是指异步任务的切换由操作语言完成，且使用单线程或较少的线程。这种能随时停止并重新激活执行的子例程(或者干脆理解为子函数) 成为**协程(coroutine)**。协程使得程序级的异步实现成为了可能。

下面我用`python3.7`中的简单例子做一说明

```python
async def task_A():
    await asyncio.sleep(1)
    print("task_A finished!")
    
async def task_B():
    await asyncio.sleep(1)
    print("task_B finished!")

asyncio.gather(*[task_A(), task_B()])
```

上述的代码会同时执行两个协程任务，看起来就像是开了多线程一样，但他们的确是在一个线程中执行。要说明的就是`await` 关键字，他只能出现在协程中，他表示当前协程即将要等待，可以让出CPU了，此时asyncio维护的事件循环会执行其他协程，每个await后的任务完成后，事件循环会感知，协程会在此处恢复继续执行。由此要说明的是仅仅依靠`async\await` 只能实现协程，不一定能实现并行。

关于asyncio的底层原理，有一篇文章[4] 写的很精彩，大家可以详细阅读，结论就是asyncio使用了操作系统自身的epoll机制维护事件循环。

`await` 表示接下来等待的就是阻塞的任务，任务完成后await后续的程序才会接着执行。`asyncio` 充当事件循环的角色。如果这两个函数是这样的：

```python
import time
def task_A():
    time.sleep(1)
    print("task_A finished!")
    
 def task_B():
    time.sleep(1)
    print("task_B finished!")
```

则除了使用多线程没办法将两个任务并行。

目前大家应该能理清楚阻塞、异步、协程的关系。其实搞不清楚也没太大关系，我们主要掌握的技能是识别需要异步的场景，并选用合适的技术解决他。正如上文所说，多线程是最直观也是大多数人都掌握的方案，也是目前工业界普遍使用的方案，如果异步的需求不多，多线程足以满足需要。但程序级别的实现是有性能优势的，下文组要介绍python语言和go语言是如何实现的。

## asyncio与goroutine实现场景

### 场景简述

我设置了一个在后端RestAPI很常见的一个业务场景: "耗时任务管理"。这个需求对于后端的朋友们一定不陌生，他包含的需求有：

- POST 创建任务，返回任务ID
- GET 查询任务的完成度、结果、开始时间、结束时间
- PUT 修改任务的状态

异步的场景主要在第一步，创建任务后接口需要立马返回任务ID，而任务放在后台执行，需要能监测到他的执行进度。

### asyncio 方案

为什么不说是python方案？因为python不止asyncio一种异步方案，在此之前已经有其他方案。asyncio是目前官方推崇的一种，希望接下来python的异步生态圈以他为核心展开。

下面就是核心POST方法，使用的库含有`aihttp`, `tortoise`

```python
@routes.post('/task')
@swagger_path("tasks.yaml")
async def create_task(request):
  	# 等待请求数据的接收
    data = await request.json()
    task_name = data.get('task_name', '')
    timestamp = datetime.datetime.now()
    # 等待连接数据库
    await init_db()
    # 数据库异步上下文
    async with in_transaction() as conn:
        task = Task(task_name=task_name, start_time=timestamp)
        # 等待数据写入
        await task.save()
        task_id = task.id
    # 异步执行任务
    asyncio.create_task(perform_task(task_id)) 
    # 当前返回
    return web.json_response({"status": "ok", "task_id": task_id})
  
 async def perform_task(task_id):
    """simple task
    """
    await asyncio.sleep(3)
    await update_status(task_id, "30%")
    await asyncio.sleep(3)
    await update_status(task_id, "60%")
    await asyncio.sleep(10)
    await update_status(task_id, "100%")
    await init_db()
    async with in_transaction() as conn:
        end_time = datetime.datetime.now()
        await Task.filter(id=task_id).using_db(conn).update(end_time=end_time)
    print(f'Task {task_id} ended!')
    
  async def update_status(task_id, status):
    await init_db()
    await Task.filter(id=task_id).update(status=status)
 
```

代码很简单，其实只有两点是重中之中：

- await的用法，写python异步程序时刻谨记要将需要挂起执行等待的任务使用await
- 并行的方法，使用asyncio提供的Task对象

第二点Task对象有必要解释一下：

```python
# 异步执行任务
asyncio.create_task(perform_task(task_id)) 
# 当前返回
return web.json_response({"status": "ok", "task_id": task_id})
```

创建一个Task对象后，会自动注册到时间循环中，不阻塞当前的程序，这里很容易理解错写成：

```python
await perform_task(task_id)
return web.json_response({"status": "ok", "task_id": task_id})
```

这样GET方法不会立马返回ID的，他会阻塞在await这里。可能这块大家就会疑惑，await不是已经将程序的执行权交出了吗为什么还能在这里阻塞？

需要明白的是，await将执行权交由主程序执行其他协程(或可等待对象，包含coroutine, Task, Future), 而GET方法本身就是当前主程序（aiohttp server程序）的一个子协程，其内的await一定是阻塞的。

这里当然也不要把`Task`对象理解为非阻塞的，如果写

```python
await syncio.create_task(perform_task(task_id)) 
```

他照样是阻塞的。

总结下`asyncio` 方案需要注意的地方：

- await 一个可等待对象。等待后面的程序会阻塞，如果等待一个非可等到对象，程序会解释错误。
- 不要期望await 能帮住你实现并行，await只能帮你实现协程，而并行需要借助asyncio的事件循环，Task对象是asyncio提供的一种高级对象，除了自动注册执行，他还有更多功能。[1]
- 协程内使用异步实现的库，不能等待一个非异步的库方法，比如`request`, `sqlAlchemy`,标准文件读写`open` ，这些都是阻塞的、不可等待的。需要对应换成基asyncio实现的非阻塞库。

第三点也是目前asyncio限制的地方，毕竟大多数人熟悉的库都不能使用了。从我去年写那篇文章，一年时间过去了，python的aio生态圈与去年比变化不大，3.8版本也没有在这里大刀阔斧地改进，国内熟悉并使用python这一套机制的人也不是很多。似乎大家还是习惯多线程和回调机制。

此外的原因，我想可能是这种编程风格，将*回调写成同步的形式*，设计者认为这是优雅的，而也有使用者认为这是别扭的。

### goroutine方案

> 不要通过共享来通信，而要通过通信来共享。
>
> ---- Go 语言设计哲学

go语言实现同样的API我选择了`gin`框架，数据库使用了标准库`database/sql` ;

跟python不同，go语言本身从设计之初就支持异步，`goroutine` 是一套语言标准机制，所以他不存在阻塞库与异步方法不兼容的问题。goroutine的原理和使用方法可以详细参考[5] 。

按照我的理解，我认为`goroutine` 既是多线程+异步的方式，如果在`asyncio` 上再加一层多线程，差不多性能上能相同。但更多的是使用方法上的不同：

```go
func main() {
	r := gin.Default()
	r.GET("/task", fetchTask)
	r.POST("/task", createTask)
	r.Run(":8080")
}
func createTask(c *gin.Context) {
	db, _ := sql.Open("mysql", DB)
	defer db.Close()
	taskName := c.DefaultPostForm("task_name", "default")
	res, _ := db.Exec("insert into task(task_name, start_time) values(? ,?)", taskName, time.Now())
	task_id, _ := res.LastInsertId()
  // 执行任务
	go func (task_id int64) {
		db, _ := sql.Open("mysql", DB)
		defer db.Close()
		fmt.Println("Work.....")
		time.Sleep(1e8)
		_, err := db.Exec("update task set status='50%' where id=?", task_id)
		if err != nil {
			fmt.Println("Error......", err)
		}
		time.Sleep(5e8)
		fmt.Println("Work.....")
		db.Exec("update task set status='100%' where id=?", task_id)
		db.Exec("update task set end_time=? where id=?", time.Now(), task_id)
	}(task_id)
  // 直接返回，goroutine非阻塞
	c.JSON(200, gin.H{
		"status": "ok",
		"task_id": task_id,
	})
}
```

与python的主要区别是两个关键字`go` 和`await`， await后的程序是阻塞的，go func() 会切换出一个goroutine执行任务，不阻塞当前程序。从逻辑上看，goroutine与多线程一样，区别在于**变量的通信和同步方式** (示例场景不涉及变量的通信)

await后阻塞程序的好处在于这迫使我们写出的程序是单线程的，数据的同步和共享变得很自然。缺点在于事件的并行是非抢占式的，他需要await来显示地"让出"cpu核心，不太符合我们一般的思维。我们总是期望计算机自行帮助我们调度事件。在之后的工作中，我也多的试着将使用多线程完成的场景改用asyncio实现，可能真正的熟悉这套用法后会体会到他本身的优雅。

goroutine的优势在于更符合我们熟悉的多线程编程思维，而数据的同步和共享可以通过channel机制很好地解决。缺点在于goroutine是半抢占式的，他使用哪个核心，是否切换线程这都是用户无法控制的。

以上两个API完整的功能代码可以从这里找到；

个人观点，正是go语言自然的设计使得为更多开发者接受，而python的异步因为历史原因虽然一直在努力但仍未大家广泛接受；我曾试图给其他人解释goroutine的用法，就按照多线程来理解很容易被接受；而python的`async/await`关键字要彻底能够清除还要从`yield\yield from`慢慢解释。

但我认为最重要的不是具体的实现方法，而是异步的思维。如何使得自己要写的任务逻辑异步且并发、高效地使用所有核心，不放任程序阻塞在IO任务，当然这往往比写同步的逻辑需要多耗费时间和脑力。

> 最后扯点别的
>
> 《大象希形》，这是我在本科阶段学习UML课程时所使用的的教材，因为所讲理论过分抽象，这门课也一致被我们认为最具玄幻色彩的课程。他一度让我怀疑软件理论世界是一门艺术而非科学。最后一次课老师关于"器"与"术"关系的探讨深刻影响了我的价值观。
>
> 学习CS至今，从编程语言到框架、软件。我渐渐明白技术栈永远是最不重要的。这也是我放弃就业而读研的原因，回望四年本科，接触的工具很多，似乎真正的道理还没参透，着实惭愧。

## Reference

- [1-asyncio-Task对象的官方文档](https://docs.python.org/3/library/asyncio-task.html)

- [2-tortoise: asyncio实现的数据库ORM](https://tortoise-orm.readthedocs.io/en/latest/examples/basic.html)

- [3-aiohttp: 异步python-http库](https://docs.aiohttp.org/en/stable/web_reference.html)

- [4- 深入理解 Python 异步编程（上）](https://mp.weixin.qq.com/s/GgamzHPyZuSg45LoJKsofA)

- [5-goroutine的本质](https://zhuanlan.zhihu.com/p/60613088)

  