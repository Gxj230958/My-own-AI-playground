import requests
import json
from datetime import datetime

WEATHER_API_KEY = "c37dd77dd41d4e6fbe750345250905"
AMAP_API_KEY = "d6b2690e21ac912688d786ad92fbfd6d"

def get_time(parameters):
  current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  return f"当前时间是 {current_time}"


def get_weather(parameters):
  try:
    url = f"http://api.weatherapi.com/v1/current.json"
    req_params = {
      "key": WEATHER_API_KEY,
      "q": parameters["location"],
      "aqi": "no"
    }
    response = requests.get(url=url, params=req_params)
    if response.status_code == 200:
      data = response.json()
      # 转换为人类可读的文本
      location = data.get("location", {}).get("name", parameters["location"])
      temp_c = data.get("current", {}).get("temp_c", "未知")
      condition = data.get("current", {}).get("condition", {}).get("text", "未知")
      
      return f"{location}当前天气：{condition}，温度{temp_c}°C"
    else:
      raise Exception("weatherapi 请求失败")
  except KeyError:
    return "缺失函数参数，请提供所有要求参数后重试"
  except Exception as e:
    print(e)
    return f"获取{parameters.get('location', '所在地')}的天气信息失败，请重试"


def get_coordinates_from_address(parameters):
  try:
    url = f"https://restapi.amap.com/v3/geocode/geo?parameters"
    req_params = {
      "key": AMAP_API_KEY,
      "address": parameters["address"],
      "output": "JSON"
    }
    response = requests.get(url=url, params=req_params)
    if response.status_code == 200:
      data = response.json()
      # 转换为人类可读的文本
      if data.get("status") == "1" and data.get("geocodes") and len(data["geocodes"]) > 0:
        address = data["geocodes"][0].get("formatted_address", parameters["address"])
        location = data["geocodes"][0].get("location", "未知")
        return f"{address}的坐标是：{location}"
      else:
        return f"未能找到{parameters['address']}的准确坐标信息"
    else:
      raise Exception("amap 请求地址经纬度失败")
  except Exception as e:
    print(e)
    return f"获取{parameters.get('address', '地址')}的位置坐标失败，请重试"
  

def get_walking_route_planning(parameters):
  try:
    url = f"https://restapi.amap.com/v3/direction/walking?parameters"
    req_params = {
      "key": AMAP_API_KEY,
      "origin": parameters["source"],
      "destination": parameters["destination"]
    }
    response = requests.get(url=url, params=req_params)
    if response.status_code == 200:
      data = response.json()
      # 转换为人类可读的文本
      if data.get("status") == "1" and data.get("route") and data["route"].get("paths"):
        path = data["route"]["paths"][0]
        distance = int(path.get("distance", 0))
        duration = int(path.get("duration", 0))
        
        # 格式化距离和时间
        if distance >= 1000:
          distance_str = f"{distance/1000:.1f}公里"
        else:
          distance_str = f"{distance}米"
          
        minutes = duration // 60
        seconds = duration % 60
        duration_str = f"{minutes}分钟" if minutes > 0 else f"{seconds}秒"
        
        return f"步行路线规划：全程约{distance_str}，预计步行时间{duration_str}。"
      else:
        return "未能找到合适的步行路线"
    else:
      raise Exception("amap 请求步行路径规划失败")
  except Exception as e:
    print(e)
    return "获取步行路径规划失败，请重试"


def get_public_transportation_route_planning(parameters):
  try:
    url = f"https://restapi.amap.com/v3/direction/transit/integrated?parameters"
    req_params = {
      "key": AMAP_API_KEY,
      "origin": parameters["source"],
      "destination": parameters["destination"],
      "city": parameters["city"]
    }
    response = requests.get(url=url, params=req_params)
    if response.status_code == 200:
      data = response.json()
      # 转换为人类可读的文本
      if data.get("status") == "1" and data.get("route") and data["route"].get("transits"):
        transit = data["route"]["transits"][0]
        distance = int(transit.get("distance", 0))
        duration = int(transit.get("duration", 0))
        
        # 格式化距离和时间
        if distance >= 1000:
          distance_str = f"{distance/1000:.1f}公里"
        else:
          distance_str = f"{distance}米"
          
        minutes = duration // 60
        hours = minutes // 60
        minutes = minutes % 60
        
        if hours > 0:
          duration_str = f"{hours}小时{minutes}分钟"
        else:
          duration_str = f"{minutes}分钟"
        
        return f"公共交通路线规划：全程约{distance_str}，预计耗时{duration_str}。"
      else:
        return "未能找到合适的公共交通路线"
    else:
      raise Exception("amap 请求公共交通路径规划失败")
  except Exception as e:
    print(e)
    return "获取公共交通路径规划失败，请重试"
  

def get_drive_route_planning(parameters):
  try:
    url = f"https://restapi.amap.com/v3/direction/driving?parameters"
    req_params = {
      "key": AMAP_API_KEY,
      "origin": parameters["source"],
      "destination": parameters["destination"],
    }
    response = requests.get(url=url, params=req_params)
    if response.status_code == 200:
      data = response.json()
      # 转换为人类可读的文本
      if data.get("status") == "1" and data.get("route") and data["route"].get("paths"):
        path = data["route"]["paths"][0]
        distance = int(path.get("distance", 0))
        duration = int(path.get("duration", 0))
        
        # 格式化距离和时间
        if distance >= 1000:
          distance_str = f"{distance/1000:.1f}公里"
        else:
          distance_str = f"{distance}米"
          
        minutes = duration // 60
        hours = minutes // 60
        minutes = minutes % 60
        
        if hours > 0:
          duration_str = f"{hours}小时{minutes}分钟"
        else:
          duration_str = f"{minutes}分钟"
        
        return f"驾车路线规划：全程约{distance_str}，预计驾车时间{duration_str}。"
      else:
        return "未能找到合适的驾车路线"
    else:
      raise Exception("amap 请求驾车路径规划失败")
  except Exception as e:
    print(e)
    return "获取驾车路径规划失败，请重试"


def get_bicycling_route_planning(parameters):
  try:
    url = f"https://restapi.amap.com/v4/direction/bicycling?parameters"
    req_params = {
      "key": AMAP_API_KEY,
      "origin": parameters["source"],
      "destination": parameters["destination"],
    }
    response = requests.get(url=url, params=req_params)
    if response.status_code == 200:
      data = response.json()
      # 转换为人类可读的文本
      if data.get("status") == "1" and data.get("data") and data["data"].get("paths"):
        path = data["data"]["paths"][0]
        distance = int(path.get("distance", 0))
        duration = int(path.get("duration", 0))
        
        # 格式化距离和时间
        if distance >= 1000:
          distance_str = f"{distance/1000:.1f}公里"
        else:
          distance_str = f"{distance}米"
          
        minutes = duration // 60
        hours = minutes // 60
        minutes = minutes % 60
        
        if hours > 0:
          duration_str = f"{hours}小时{minutes}分钟"
        else:
          duration_str = f"{minutes}分钟"
        
        return f"骑行路线规划：全程约{distance_str}，预计骑行时间{duration_str}。"
      else:
        return "未能找到合适的骑行路线"
    else:
      raise Exception("amap 请求骑行路径规划失败")
  except Exception as e:
    print(e)
    return "获取骑行路径规划失败，请重试"