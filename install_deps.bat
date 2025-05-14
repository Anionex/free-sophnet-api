@echo off
echo 安装 OpenAI Chat Proxy 所需的依赖库...
pip install fastapi uvicorn aiohttp pydantic loguru orjson pyyaml
echo 依赖安装完成!
pause 