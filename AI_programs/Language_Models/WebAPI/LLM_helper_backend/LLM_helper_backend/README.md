# LLM_helper_backend
### 安装依赖库：在终端中输入以下命令
```shell
cd llm_helper_backend
pip install -r requirements.txt
```
### 代码结构
```shell
.
├── utils               # 该package管理各种工具函数、类
│   ├── lifespan.py     # 管理启动后端的初始化和退出代码
│   ├── error.py        # 注册了一些异常处理函数
│   ├── decoration.py   # API的成功返回格式
│   ├── config.py       # 读取配置文件
│   └── __init__.py     
├── services            # 该package管理实验2中的各种辅助功能
│   └── __init__.py     
├── routers             # 该package管理各API的路由以及内部逻辑
│   ├── simple_llm.py   # 一个简单的llm示例路由代码
│   └── __init__.py     
├── requirements.txt    # 依赖库
├── main.py             # 后端入口，在这里调用注册路由router的函数
├── .env                # 配置参数
└── README.md
```
### 运行
```shell
➜ ~ python3 main.py
INFO:     Started server process [8436]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
...
➜ ~ curl -X 'POST' \
  'http://127.0.0.1:8000/api/simple_llm/response' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "你好"
}'
{"code":0,"data":{"answer":"你好呀！有什么可以帮你的吗？😊"}}
```