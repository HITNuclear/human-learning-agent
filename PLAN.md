# DFS Learning Agent — 项目架构计划

## 项目背景

传统线性对话无法保留上下文层级，导致深入追问时知识脉络断裂。本系统将每次问答保存为树的节点，用户可以从任意答案片段"向下钻取"，每次问答自动将祖先节点的问题+答案作为上下文传入 LLM，最终形成可持久化、可回溯的知识树。

---

## 技术栈

| 层 | 技术 | 理由 |
|---|---|---|
| 前端 | Next.js 14 + TypeScript + Tailwind + shadcn/ui | 路由、流式渲染、组件库一体化 |
| 状态 | Zustand + TanStack React Query | 轻量全局状态 + 服务端状态缓存 |
| 后端 | Python FastAPI + uvicorn | async、SSE 原生支持 |
| 数据库 | SQLite (aiosqlite + SQLAlchemy 2.0 async) | 零运维、本地持久化 |
| 向量库 | ChromaDB (本地 PersistentClient) | 无需外部服务 |
| 嵌入模型 | sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` | 支持中文 + 50 语言，CPU 可跑 |
| LLM | DeepSeek API (openai SDK, base_url=https://api.deepseek.com) | OpenAI 兼容格式，支持前缀缓存 |
| RAG 来源 | pypdf + beautifulsoup4/httpx + 直接文本 | 三种摄入路径 |
| 流式传输 | sse-starlette (服务端) + @microsoft/fetch-event-source (客户端) | 稳定 SSE，支持断线处理 |

---

## 核心数据模型

```python
class Tree(Base):
    id, title, created_at

class Node(Base):
    id, tree_id, parent_id   # parent_id=None 表示根节点
    selected_text             # 用户从父答案选取的文字片段
    question                  # 用户问题
    answer                    # AI 回答（为空时触发流式生成）
    depth                     # 缓存深度，避免递归查询
    created_at
```

---

## 项目结构

```
human-learning-agent/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + lifespan
│   │   ├── config.py            # pydantic-settings (.env)
│   │   ├── database.py          # async engine, Base, get_db
│   │   ├── models/tree.py       # SQLAlchemy ORM models
│   │   ├── models/node.py
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── api/trees.py         # GET/POST /api/trees
│   │   ├── api/nodes.py         # POST /api/nodes + SSE stream
│   │   ├── api/rag.py           # upload/text/url endpoints
│   │   └── services/
│   │       ├── llm_service.py       # DeepSeek stream_completion()
│   │       ├── context_builder.py   # 祖先路径 → LLM messages
│   │       ├── rag_service.py       # ChromaDB + SentenceTransformer singleton
│   │       └── document_processor.py  # PDF/text/URL → chunks
│   ├── data/db.sqlite3
│   ├── data/chroma/
│   ├── uploads/
│   └── requirements.txt
├── frontend/
│   ├── app/
│   │   ├── page.tsx             # 树列表首页
│   │   └── tree/[treeId]/
│   │       ├── layout.tsx       # 侧边栏 + 内容区布局
│   │       └── node/[nodeId]/page.tsx  # 节点视图
│   ├── components/
│   │   ├── tree/TreeSidebar.tsx     # 递归树节点列表
│   │   ├── node/NodeView.tsx        # 问答展示主组件
│   │   ├── node/AnswerRenderer.tsx  # Markdown + 代码高亮
│   │   ├── node/InlineAskPanel.tsx  # 内联追问面板（紧贴选中文字）
│   │   ├── node/Breadcrumb.tsx      # 根 → 当前节点路径
│   │   └── rag/RagPanel.tsx         # 知识库管理
│   ├── hooks/
│   │   ├── useStreamAnswer.ts   # SSE 客户端 hook
│   │   └── useTextSelection.ts  # 鼠标选取文本 hook
│   ├── lib/api.ts               # fetch 类型化 API 客户端
│   ├── lib/store.ts             # Zustand store
│   └── next.config.ts           # rewrites /api/* → FastAPI
```

---

## 关键设计：上下文构建 (context_builder.py)

```
消息列表构造（depth N 节点）：
[system: "学习助手 + RAG chunks"]
[user: root.question]
[assistant: root.answer]
[user: "Based on: '{A.selected_text}', I want to understand: A.question"]
[assistant: A.answer]
...
[user: "Based on: '{current.selected_text}', I want to understand: current.question"]
```

祖先内容不变 → DeepSeek 前缀缓存命中 → 成本随深度增加缓慢。

---

## 核心 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | /api/trees | 树列表 |
| POST | /api/trees | 创建树（同时创建根节点） |
| GET | /api/trees/{id}/nodes | 获取树所有节点（flat list） |
| POST | /api/nodes | 创建子节点（answer="" 待填充） |
| GET | /api/nodes/{id}/stream | SSE 流式生成并保存 answer |
| POST | /api/rag/upload | PDF 上传 |
| POST | /api/rag/text | 粘贴文本 |
| POST | /api/rag/url | 爬取网页 |
| GET | /api/rag/sources | 知识源列表 |

**两阶段节点创建设计**（关键决策）：
POST 同步返回 node（含 id） → 前端立即获取节点 ID → GET /stream 异步流式填充 answer。
好处：URL 可立即反映新节点，stream 端点幂等（answer 非空则直接返回已有内容）。

---

## RAG 设计

- 单 ChromaDB collection，metadata 含 `tree_id`（"global" 或具体 tree id）
- 查询时 where filter：`tree_id IN [current_tree_id, "global"]`
- SentenceTransformer 在应用启动时作为单例初始化（~3s，~500MB RAM）
- 文本切分：按段落优先，再按字符数滑动窗口（chunk_size=512, overlap=64）

---

## UI 交互流程

```
用户在答案中高亮文字
  → useTextSelection hook 捕获选取内容 + 位置
  → 浮动"继续追问"按钮出现（position: fixed，紧贴选中文字下方）
  → 点击 → InlineAskPanel 在原位弹出
      阶段一：小输入框（300px）
        → 用户输入问题，Enter 提交
        → POST /api/nodes 创建子节点
      阶段二：回答面板（420px）
        → GET /stream 开始 SSE 流式输出
        → useStreamAnswer 逐 token 更新 Zustand
        → AnswerRenderer 实时渲染 Markdown
        → 双击面板 → router.push 跳转完整节点页面
```

树侧边栏：递归 React 组件（非 D3），CSS `border-left` 绘制层级线，active 节点高亮。

---

## 重要实现细节

### SSE 流式传输关键修复

1. **DB session 问题**：FastAPI 依赖注入的 `get_db` session 在 endpoint 函数返回时关闭，SSE generator 内无法使用。解决：在 generator 内部用 `async with AsyncSessionLocal() as save_db` 单独开 session 保存 answer。

2. **fetchEventSource 重连问题**：SSE 流正常结束后库会自动重试。解决：在 "done"/"full"/"error" 事件中调用 `ctrl.abort()`。

3. **React 批处理问题**：React 18 自动批处理导致 token 逐个更新被合并渲染。解决：用 `flushSync(() => appendToken(ev.data))` 强制同步渲染。

4. **Next.js 代理缓冲**：`next.config.ts` 的 `rewrites` 会缓冲整个响应。解决：前端直接请求后端 `window.location.hostname:8000`，绕过 Next.js 代理。

5. **Tab 切换断流**：切换标签页时 fetchEventSource 断开连接。解决：设置 `openWhenHidden: true`。

### 依赖版本约束

```
httpx==0.27.2   # 0.28.0 破坏了 chromadb (_state AttributeError) 和 openai SDK (proxies 参数移除)
```

---

## 实施顺序

1. **后端基础**：FastAPI app、models、CRUD API（无 LLM）
2. **LLM 集成**：context_builder + llm_service + SSE 端点
3. **前端骨架**：Next.js 初始化、路由、静态节点视图、API 代理
4. **核心 UX**：流式渲染 + 文字选取 + InlineAskPanel + 完整 DFS 流程
5. **RAG**：ChromaDB + 三种文档摄入 + 接入流式端点
6. **打磨**：动画、加载态、错误处理、移动端适配

---

## 验证方案

1. `curl -X POST localhost:8000/api/trees` 创建树，检查 DB 中 root node
2. `curl --no-buffer localhost:8000/api/nodes/{id}/stream` 验证 SSE token 流
3. 前端：创建树 → 首页显示 → 进入节点视图 → 高亮文字 → 提交子问题 → 侧边栏新增子节点
4. RAG：上传 PDF → `curl localhost:8000/api/rag/sources` 确认入库 → 新建问题验证 RAG chunks 进入 context
5. 深度 3+ 的节点：检查 messages 列表包含完整祖先链

---

## InlineAskPanel 设计（已实现）

替代原来的右侧 DrillDownSheet，改为紧贴选中文字的内联面板。

### 两阶段状态

**阶段一（nodeId === null）：输入框**
- `position: fixed`，top = `rect.bottom + 8`，left = `rect.left`（clamp 视口边界）
- 宽 300px，显示选中文本摘要 + textarea
- Enter 提交，Shift+Enter 换行，Esc 关闭
- 提交 → `api.createNode` → setNodeId → startStream(newNode.id)

**阶段二（nodeId !== null）：回答面板**
- 宽 420px，maxHeight 320px 内容区可滚动
- 顶栏：问题文字 + "双击打开完整页面"提示 + X 关闭
- 内容：streaming 时显示 `streaming.content`，完成后显示 React Query 缓存
- `onDoubleClick` → router.push 跳转完整节点页面 + onClose()
