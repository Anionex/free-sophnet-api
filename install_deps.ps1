Write-Host "安装 OpenAI Chat Proxy 所需的依赖库..." -ForegroundColor Green

try {
    pip install fastapi uvicorn aiohttp pydantic loguru orjson pyyaml
    Write-Host "依赖安装完成!" -ForegroundColor Green
} catch {
    Write-Host "安装依赖失败: $_" -ForegroundColor Red
}

Read-Host "按Enter键继续" 