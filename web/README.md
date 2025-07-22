# LangChain AI SDK Adapter Web Demo

这是一个完整的 FastAPI + Vue.js 示例，展示了如何使用 LangChain AI SDK Adapter 进行流式聊天。该项目实现了"实例隔离 + 上下文管理器"方案，确保线程安全和调用简单。

## 项目结构

```
web/
├── backend/                 # FastAPI 后端
│   ├── main.py             # 主应用文件 - FastAPI 路由和端点
│   ├── models.py           # 数据模型 - Pydantic 模型定义
│   ├── agents.py           # LangChain Agent 配置和工具
│   ├── stream_handlers.py  # 流处理器 - 自动/手动模式实现
│   └── requirements.txt    # Python 依赖包列表
├── frontend/               # Vue.js 前端
│   ├── src/
│   │   ├── App.vue        # 主组件 - 聊天界面
│   │   ├── main.js        # 入口文件
│   │   └── style.css      # 全局样式
│   ├── package.json       # Node.js 依赖和脚本
│   ├── vite.config.js     # Vite 构建配置
│   └── index.html         # HTML 模板
├── start_backend.py       # 后端启动脚本 (推荐)
├── start_frontend.js      # 前端启动脚本 (推荐)
└── README.md             # 项目说明文档
```

## 快速开始

### 方式一：使用启动脚本 (推荐)

#### 启动后端
```bash
# 在 web 目录下运行
python start_backend.py
```

启动脚本会自动：
- 检查 Python 版本 (需要 3.8+)
- 检查并提示安装缺失的依赖
- 设置环境变量
- 启动 FastAPI 服务器 (http://localhost:8000)

#### 启动前端
```bash
# 在 web 目录下运行
node start_frontend.js
```

启动脚本会自动：
- 检查 Node.js 版本 (需要 18+)
- 检查并安装缺失的依赖
- 检查后端服务状态
- 启动 Vue.js 开发服务器 (http://localhost:3000)

### 方式二：手动启动

#### 启动后端

1. 安装 Python 依赖：
```bash
cd backend
pip install -r requirements.txt
```

2. 设置环境变量 (可选)：
```bash
# Windows
set OPENAI_API_KEY=your-api-key-here

# Linux/Mac
export OPENAI_API_KEY=your-api-key-here
```

3. 启动后端服务：
```bash
python main.py
# 或者使用 uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 启动前端

1. 安装 Node.js 依赖：
```bash
cd frontend
npm install
# 或者使用 yarn
yarn install
```

2. 启动前端服务：
```bash
npm run dev
# 或者使用 yarn
yarn dev
```

## 访问应用

- **前端界面**: http://localhost:3000
- **后端 API**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health

## 功能特性

### 核心特性
- ✅ **线程安全**: 使用实例隔离 + 上下文管理器方案
- ✅ **流式响应**: 支持实时流式聊天体验
- ✅ **双模式支持**: 
  - **自动模式**: 使用 `to_data_stream_response` 自动处理
  - **手动模式**: 使用 LangChain 回调精确控制
- ✅ **工具调用**: 支持网络搜索、数学计算、时间查询
- ✅ **现代 UI**: 基于 Vue.js 3 + @ai-sdk/vue
- ✅ **响应式设计**: 适配桌面和移动设备

### 技术特性
- ✅ **类型安全**: 完整的 Pydantic 模型定义
- ✅ **错误处理**: 完善的错误捕获和用户提示
- ✅ **开发友好**: 热重载、详细日志、调试信息
- ✅ **CORS 支持**: 跨域请求处理
- ✅ **健康检查**: 服务状态监控
- ✅ **环境配置**: 灵活的环境变量配置

## API 端点

### 聊天端点
- `POST /api/chat/auto` - 自动模式流式聊天
- `POST /api/chat/manual` - 手动模式流式聊天
- `POST /api/chat` - 统一聊天端点 (根据参数选择模式)
- `POST /api/chat/sync` - 同步聊天 (非流式)

### 信息端点
- `GET /` - 根路径健康检查
- `GET /health` - 详细健康检查
- `GET /api/models` - 获取可用模型列表
- `GET /api/tools` - 获取可用工具列表

## 使用示例

### 基本对话
```
用户: 你好，请介绍一下自己
AI: 你好！我是一个基于 LangChain 的 AI 助手...
```

### 工具调用示例
```
用户: 搜索一下今天的天气
AI: [调用搜索工具] 根据搜索结果，今天的天气是...

用户: 计算 123 * 456
AI: [调用计算工具] 123 * 456 = 56088

用户: 现在几点了？
AI: [调用时间工具] 现在是 2024年1月1日 12:00:00
```

## 开发说明

### 后端开发
- 基于 FastAPI 框架
- 使用 LangChain 进行 Agent 管理
- 实现了完整的流式处理逻辑
- 支持多种工具集成

### 前端开发
- 基于 Vue.js 3 Composition API
- 使用 @ai-sdk/vue 处理流式响应
- Vite 构建工具，支持热重载
- 响应式 UI 设计

### 自定义扩展
1. **添加新工具**: 在 `backend/agents.py` 中定义新的工具函数
2. **修改 UI**: 编辑 `frontend/src/App.vue` 组件
3. **添加新端点**: 在 `backend/main.py` 中添加新的路由
4. **自定义样式**: 修改 `frontend/src/style.css`

## 故障排除

### 常见问题

1. **后端启动失败**
   - 检查 Python 版本 (需要 3.8+)
   - 确保所有依赖已安装
   - 检查端口 8000 是否被占用

2. **前端启动失败**
   - 检查 Node.js 版本 (需要 18+)
   - 删除 `node_modules` 重新安装依赖
   - 检查端口 3000 是否被占用

3. **API 调用失败**
   - 确保后端服务正在运行
   - 检查 CORS 配置
   - 查看浏览器控制台错误信息

4. **工具调用失败**
   - 检查 OpenAI API Key 是否正确设置
   - 查看后端日志获取详细错误信息

### 日志查看
- 后端日志: 在启动后端的终端中查看
- 前端日志: 在浏览器开发者工具的控制台中查看

## 许可证

本项目仅用于演示和学习目的。