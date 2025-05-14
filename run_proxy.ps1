Write-Host "启动 OpenAI Chat Proxy 服务..." -ForegroundColor Green

try {
    python openai_chat_proxy.py
} catch {
    Write-Host "启动失败: $_" -ForegroundColor Red
    Read-Host "按Enter键退出"
    exit 1
}

Read-Host "按Enter键退出" 