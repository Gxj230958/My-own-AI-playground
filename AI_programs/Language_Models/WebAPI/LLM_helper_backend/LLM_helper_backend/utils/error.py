from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException
from pydantic.error_wrappers import ValidationError
from fastapi.exceptions import FastAPIError, RequestValidationError


def register_error(app: FastAPI):
    """
    注册异常处理
    """
    @app.exception_handler(ApiException)
    async def api_exception_handler(request: Request, exc: ApiException):
        """
        api类error raise 封装
        {'code': 自定义error code, 'data': error解释消息, 'type': 'API Error'}
        """
        return JSONResponse(
            content=exc.dict(type='API Error')
        )

    @app.exception_handler(HTTPException)
    async def starlette_exception_handler(request: Request, exc: HTTPException):
        """
        http类error raise 重构，确保全局error一致性
        example:
        404时控制台本身不报错(返回200)，一切由错误码来体现，方便前端做error跳转
        {'code': http状态码, 'data': error解释消息, 'type': 'HTTP Error'}
        """
        return JSONResponse(
            content={'code': exc.status_code, 'data': exc.detail, 'type': 'HTTP Error'}
        )

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        return JSONResponse(
            content={'code': 500, 'data': f'{exc.errors()}', 'type': 'Validation Error'}
        )

    @app.exception_handler(FastAPIError)
    async def runtime_exception_handler(request, exc: FastAPIError):
        return JSONResponse(
            content={'code': 500, 'data': exc.__doc__, 'type': 'Runtime Error'}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc: RequestValidationError):
        return JSONResponse(
            content={'code': 500, 'data': str(exc), 'type': 'value_error.missing'}
        )

class ApiException(Exception):

    uni_name = "ApiException"  # 用户使用api可能遇到的错误码

    def __init__(self, code, reason=None):
        self.code = code
        self.reason = reason

    def dict(self, **kwargs):
        result = {
            "code": self.code,
            "data": self.reason
        }
        result.update(kwargs)
        return result

    def __call__(self, *, code=None, reason=None):
        return type(self)(code or self.code, reason or self.reason)

    def __str__(self):
        return f'Code: {self.code}, reason: {self.reason}'