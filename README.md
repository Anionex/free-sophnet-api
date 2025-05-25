# Free SophNet API 代理

> **声明：**  
> 仅供网络安全学习交流使用
> 请在24小时内删除，造成的任何后果与项目开发者无关

这是一个轻量级的 API 代理服务，专注于转发请求到 SophNet API 服务。支持流式响应输出，API 密钥映射，自动获取匿名 token，以及 IP 访问控制。

## 功能特点

- 兼容 OpenAI `/v1/chat/completions` 接口格式
- 支持流式输出 (SSE)
- 支持自动获取 SophNet 匿名 token
- 支持自定义转发目标路径
- 支持请求字段别名映射（如 model 到 model_id）
- 支持 OpenAI Function Calling 与 Tools 功能
- API 密钥映射，便于管理和保护实际 API 密钥
- IP 白名单/黑名单访问控制
- 可配置的请求和响应日志记录
- 支持 HTTP 代理
- 健康检查 API
- 支持获取模型列表
- 该接口也支持function call/tool call功能

## 快速开始
### 小白向快速启动脚本
（linux需要安装docker compose，windows需要先安装docker desktop）
```
git clone https://github.com/Anionex/free-sophnet-api
cd free-sophnet-api
cp config.yml.example config.yml
docker compose up -d
```
### 从源码开始启动（进阶向）
```
git clone https://github.com/Anionex/free-sophnet-api
cd free-sophnet-api
```
1. 复制示例配置文件并根据需要修改：

   ```
   cp config.yml.example config.yml
   ```

2. 编辑 `config.yml` 文件，设置你的配置参数(也可以直接全部保持默认)。

### 依赖库

使用项目的 pyproject.toml 安装:

```
pip install -e .
```

✨推荐使用 `uv`:

```
uv sync # 环境安装
uv run soph_forward # 运行
```

### 启动服务

由于 uvloop 依赖不支持 windows, 故推荐在 linux 上启动:

```
python -m src.main
```

服务默认在 `0.0.0.0:8000` 上启动，可以通过修改代码中的端口号来更改。

### 启动服务(docker, ✨推荐)

(可选步骤):
1. 在 `docker-compose.yml` 中添加 `restart: always` 以开机自启动.
2. 编辑 `ports` 端口映射 (服务在容器内监听 `8000` 端口).

启动:

```
docker compose up -d
```

## 使用方法

服务启动后，可以将原本发送到 OpenAI API 的请求改为发送到：

```
http://localhost:8000/v1/chat/completions
```

在请求头中使用你在配置文件中设置的前端密钥（默认为sk-2jc7k79eca#）：

```
Authorization: Bearer your_frontend_key
```

服务会自动将其映射为实际的 SophNet API 密钥或自动获取匿名 token。

### 自定义转发路径

你可以在配置文件中设置 `forward_chat_path` 参数来自定义转发目标的路径。例如：

```yaml
# 将请求转发到 SophNet 特定路径
forward_chat_path: "/api/open-apis/projects/Ar79PWUQUAhjJOja2orHs/chat/completions"
```

### 字段别名映射

你可以在配置文件中设置 `field_aliases` 参数来自定义请求JSON数据中字段名的转换规则：

```yaml
field_aliases:
  "model": "model_id"  # 将请求中的model字段转换为model_id
```

使用此配置，当发送以下请求时：

```json
{
  "model": "DeepSeek-V3-Fast",
  "messages": [
    {
      "role": "user",
      "content": "你好"
    }
  ],
  "temperature": 0,
  "stream": true
}
```

服务会自动将其转换为以下格式后再转发：

```json
{
  "model_id": "DeepSeek-V3-Fast",
  "messages": [
    {
      "role": "user",
      "content": "你好"
    }
  ],
  "temperature": 0,
  "stream": true
}
```

字段别名功能会递归应用到嵌套的对象和数组中，非常适合处理复杂的JSON结构。

### 请求示例

使用 curl 发送请求：

```powershell
# powershell
curl -X POST http://localhost:8000/v1/chat/completions `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer your_frontend_key" `
  -d '{
    "model": "DeepSeek-V3-Fast",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

## 健康检查

可以通过访问 `/health` 端点来检查服务是否正常运行：

```
http://localhost:8000/health
```

## 获取模型列表

可以通过访问 `/v1/models` 端点查询可用的模型：

```
http://localhost:8000/v1/models
```

## 如何获得更好的性能？

研究发现性能瓶颈在于官网对/chat/completions接口的rate limit。因此解决方案是使用代理池发出请求以尽量减少被限制的次数。

在config.yml中配置代理：

```
# 代理服务器URL(如果需要)
# 示例: socks5://127.0.0.1:7890 推荐使用socks5
```

同时，在config.yml中设置参数：

```
keepalive_timeout: 0
force_close: true
```

当配置了proxy，脚本会自动将请求头中的 `x-real-ip` 等字段统一设置为当前发起请求的IP地址。虽然实测没有太多差别；

压测命令(evalscope)：

```bash
uv run tests/bench.py --host http://localhost:8000
# 或
python tests/bench.py --host http://localhost:8000
```

> [!NOTE]
> 需要主程序正在运行, 上面的 `--host` 参数设置为主程序绑定的地址.

压测结果:

| 项目                       | 优化前    | 优化后    |
| :------------------------- | :-------- | :-------- |
| 并发数                     | 16        | 16        |
| 总请求数                   | 100       | 100       |
| 成功请求数                 | 42        | 88        |
| 失败请求数                 | 58        | 12        |
| 输出吞吐量 (tok/s)         | 797.8415  | 951.4029  |
| 总吞吐量 (tok/s)           | 827.2852  | 992.081   |
| 请求吞吐量 (req/s)         | 1.4652    | 1.8644    |
| 平均延迟 (s)               | 7.3841    | 7.6495    |
| 平均首次 token 时间 (s)    | 0.8123    |           |
| 平均每个输出 token 时间 (s) | 0.0122    |           |
| 测试总耗时 (s)             |           | 47.1998   |

优化后详细测试结果:

| Percentiles | TTFT (s) | ITL (s) | TPOT (s) | Latency (s) | Input tokens | Output tokens | Output (tok/s) | Total (tok/s) |
| :---------- | :------- | :------ | :------- | :---------- | :------------ | :------------ | :------------- | :------------ |
| 10%         | 0.4447   | 0.0108  | 0.015    | 3.956       | 13            | 237           | 44.2754        | 48.2932       |
| 25%         | 0.4718   | 0.0293  | 0.0154   | 7.0817      | 17            | 361           | 48.9376        | 51.8933       |
| 50%         | 0.5084   | 0.0342  | 0.0174   | 8.8157      | 23            | 450           | 53.9693        | 57.393        |
| 66%         | 0.528    | 0.0363  | 0.0183   | 9.545       | 26            | 526           | 58.0365        | 61.0627       |
| 75%         | 0.5557   | 0.0391  | 0.019    | 10.0414     | 27            | 567           | 59.1532        | 62.4369       |
| 80%         | 0.5686   | 0.0435  | 0.0194   | 10.2598     | 31            | 582           | 61.1363        | 63.1524       |
| 90%         | 0.7113   | 0.0637  | 0.0205   | 10.8933     | 33            | 617           | 62.8718        | 66.3666       |
| 95%         | 0.7288   | 0.0779  | 0.0216   | 11.759      | 35            | 670           | 65.0002        | 67.1137       |
| 98%         | 0.957    | 0.1292  | 0.0225   | 13.7054     | 37            | 762           | 68.2566        | 73.3314       |
| 99%         | 0.957    | 0.1763  | 0.0225   | 13.7054     | 37            | 762           | 68.2566        | 73.3314       |

如果有更好的方法，欢迎交流！
