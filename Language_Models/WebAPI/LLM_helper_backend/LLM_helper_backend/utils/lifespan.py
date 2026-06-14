from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在web server启动时进行一些初始化工作 (初始化 llm 相关参数, etc.)
    pass
    # 添加其他函数
    yield
    # 在web server停止时进行清理工作 (关闭文件, etc.)
    pass