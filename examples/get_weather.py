"""天气查询工具，基于LangChain实现。"""

from typing import Dict, Any
import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class WeatherInput(BaseModel):
    """天气查询输入参数。"""
    latitude: float = Field(
        description="Location latitude in decimal degrees (e.g., 39.9042 for Beijing)",
        ge=-90.0,
        le=90.0,
        examples=[39.9042, 40.7128, 51.5074]
    )
    longitude: float = Field(
        description="Location longitude in decimal degrees (e.g., 116.4074 for Beijing)",
        ge=-180.0,
        le=180.0,
        examples=[116.4074, -74.0060, -0.1278]
    )


class WeatherTool(BaseTool):
    """天气查询工具 - 获取指定地理位置的当前天气信息。"""
    
    name: str = "get_weather"
    description: str = (
        "Get current weather information for a specific geographic location. "
        "Requires latitude and longitude coordinates. Returns temperature, "
        "sunrise/sunset times, and other weather data from Open-Meteo API."
    )
    args_schema: type[BaseModel] = WeatherInput
    
    async def _arun(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """异步执行天气查询。"""
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={latitude}&longitude={longitude}&"
                f"current=temperature_2m&hourly=temperature_2m&"
                f"daily=sunrise,sunset&timezone=auto"
            )
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                weather_data = response.json()
                
            return weather_data
            
        except Exception as e:
            return {
                "error": f"天气查询失败: {str(e)}",
                "latitude": latitude,
                "longitude": longitude
            }
    
    def _run(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """同步执行天气查询。"""
        import asyncio
        return asyncio.run(self._arun(latitude, longitude))