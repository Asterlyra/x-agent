# x-agent

单用户本地 Web Agent，使用 FastAPI、LangGraph 与 React，支持聊天、工具调用和有界工作流协作。会话保存在本机。

需要 Python 3.11、Node.js 和 pnpm。

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd frontend && pnpm install && cd ..
.venv/bin/python start_dev.py
```

在本地环境变量或不提交的 `.env` 中设置 `DEEPSEEK_API_KEY`，访问 http://127.0.0.1:5173。服务面向本机单用户使用。

测试：`.venv/bin/python -m pytest -q`；前端在 `frontend/` 执行 `pnpm test` 和 `pnpm build`。自动化测试主要使用模拟模型，不代表真实模型质量已经验证。

## 许可

本项目原创代码采用 [MIT License](LICENSE)。第三方代码、数据和素材各自适用其原有许可。
