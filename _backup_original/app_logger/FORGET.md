# 记忆流程:

- getLogger   注册

- setLevel   设置等级

- if handlers: return   #防止重复handler

- add FileHandler  #文件处理器

- set Formatter   #定义格式

- add ConsoleHandler（RichHandler）   #控制台处理器

- return logger   #返回实例对象
