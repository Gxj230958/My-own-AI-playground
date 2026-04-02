import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from utils import lifespan, register_error, SuccessResponse
from services import answer, answer_stream
import json

# 使用lifespan函数来完成初始化和退出处理
app = FastAPI(
    lifespan=lifespan,
    title="LLM Helper API",
    description="""
    LLM助手API，支持普通响应和流式响应。
    
    ## 使用流式响应的curl命令示例:
    ```
    curl -X 'POST' \\
      'http://127.0.0.1:8000/api/response' \\
      -H 'accept: text/event-stream' \\
      -H 'Content-Type: application/json' \\
      -d '{"question": "你好"}'
    ```
    
    ## 使用普通响应的curl命令示例:
    ```
    curl -X 'POST' \\
      'http://127.0.0.1:8000/api/response' \\
      -H 'accept: application/json' \\
      -H 'Content-Type: application/json' \\
      -d '{"question": "你好", "stream": false}'
    ```
    """
)

# 注册一些异常处理函数
register_error(app)

@app.get("/")
async def root():
    return "hello world!"

# 定义post JSON格式
class HelloRequest(BaseModel):
    name: str
    
@app.post("/hello")
async def hellowho(hello_request: HelloRequest):
    return "hello " + hello_request.name

class RequestQuestion(BaseModel):
    question: str
    stream: bool = True  # 默认使用流式响应

# 为返回包裹了一层JSON格式，未来被前端使用
@app.post("/api/response")
async def response(request_question: RequestQuestion, request: Request):
    """
    处理用户问题并返回回答
    
    - **question**: 用户问题
    - **stream**: 是否使用流式响应（默认为true）
    
    当stream=true时，返回的是SSE格式的流式响应
    当stream=false时，返回的是常规JSON响应
    """
    # 如果请求流式响应
    if request_question.stream:
        async def generate():
            for chunk in answer_stream(request_question.question):
                if chunk['content']:
                    # 使用SSE格式
                    yield f"data: {json.dumps({'code': 0, 'data': {'answer': chunk['content']}})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    # 否则使用常规响应
    else:
        ans = answer(request_question.question)
        return JSONResponse(content={"code": 0, "data": {"answer": ans}})

# 流式响应端点 (保留兼容性)
@app.post("/api/stream-response")
async def stream_response(request_question: RequestQuestion):
    """
    处理用户问题并返回流式回答（SSE格式）
    
    - **question**: 用户问题
    """
    async def generate():
        for chunk in answer_stream(request_question.question):
            if chunk['content']:
                yield f"data: {json.dumps({'code': 0, 'data': {'answer': chunk['content']}})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000
    uvicorn.run(app, host=host, port=port)