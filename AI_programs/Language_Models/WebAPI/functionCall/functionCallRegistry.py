from functionCallList import *


function_registry = {
  "get_time": get_time,
  "get_weather": get_weather,
  "get_coordinates_from_address": get_coordinates_from_address,
  "get_walking_route_planning": get_walking_route_planning,
  "get_public_transportation_route_planning": get_public_transportation_route_planning,
  "get_drive_route_planning": get_drive_route_planning,
  "get_bicycling_route_planning": get_bicycling_route_planning
}

function_desc = [
  {
    "type": "function",
    "function": {
      "name": "get_time",
      "description": "获取当前时间"
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "获取指定地点的天气信息",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "查询天气的地点，如城市名称"
          }
        },
        "required": ["location"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_coordinates_from_address",
      "description": "获取地址的经纬度坐标",
      "parameters": {
        "type": "object",
        "properties": {
          "address": {
            "type": "string",
            "description": "需要获取经纬度的地址"
          }
        },
        "required": ["address"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_walking_route_planning",
      "description": "获取步行路径规划",
      "parameters": {
        "type": "object",
        "properties": {
          "source": {
            "type": "string",
            "description": "起点的经纬度，格式为'经度,纬度'"
          },
          "destination": {
            "type": "string",
            "description": "终点的经纬度，格式为'经度,纬度'"
          }
        },
        "required": ["source", "destination"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_public_transportation_route_planning",
      "description": "获取公共交通路径规划",
      "parameters": {
        "type": "object",
        "properties": {
          "source": {
            "type": "string",
            "description": "起点的经纬度，格式为'经度,纬度'"
          },
          "destination": {
            "type": "string",
            "description": "终点的经纬度，格式为'经度,纬度'"
          },
          "city": {
            "type": "string",
            "description": "城市名称"
          }
        },
        "required": ["source", "destination", "city"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_drive_route_planning",
      "description": "获取驾车路径规划",
      "parameters": {
        "type": "object",
        "properties": {
          "source": {
            "type": "string",
            "description": "起点的经纬度，格式为'经度,纬度'"
          },
          "destination": {
            "type": "string",
            "description": "终点的经纬度，格式为'经度,纬度'"
          }
        },
        "required": ["source", "destination"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_bicycling_route_planning",
      "description": "获取骑行路径规划",
      "parameters": {
        "type": "object",
        "properties": {
          "source": {
            "type": "string",
            "description": "起点的经纬度，格式为'经度,纬度'"
          },
          "destination": {
            "type": "string",
            "description": "终点的经纬度，格式为'经度,纬度'"
          }
        },
        "required": ["source", "destination"]
      }
    }
  }
]