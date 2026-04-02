from fastapi import Response
import typing
import json


class SuccessResponse(Response):
    """
    重构JSONResponse, 在API router中设定为default response class，统一返回格式
    """
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            {'code': 0, 'data': content},
            ensure_ascii=False,
            allow_nan=False,
            indent=None,
            separators=(",", ":"),
        ).encode("utf-8")