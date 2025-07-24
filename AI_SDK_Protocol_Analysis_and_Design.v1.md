# AI SDK v4 vs v5 协议分析与双协议支持设计方案

## 概述

本文档分析了 AI SDK v4 和 v5 协议的差异，并提出了一个支持两种协议版本的设计方案。基于联网搜索和现有代码分析，我们确认了两种协议在格式、响应头和文本处理方式上的关键差异。

## AI SDK v4 协议完整规范

### Data Stream Protocol

AI SDK v4 使用自定义的数据流协议，每个流部分的格式为 `TYPE_ID:CONTENT_JSON\n`。

#### 响应头要求
- **必需头**: `x-vercel-ai-data-stream: v1`
- **内容类型**: `text/plain; charset=utf-8`
- **缓存控制**: `Cache-Control: no-cache`
- **连接**: `Connection: keep-alive`

#### 支持的流部分类型

##### 文本部分 (Text Part)
文本内容会在接收时追加到消息中。
- **格式**: `0:string\n`
- **示例**: `0:"example"\n`

##### 推理部分 (Reasoning Part)
推理内容会在接收时追加到消息中，通过 `reasoning` 属性访问。
- **格式**: `g:string\n`
- **示例**: `g:"I will open the conversation with witty banter."\n`

##### 编辑推理部分 (Redacted Reasoning Part)
包含已编辑的推理数据，通过 `redacted_reasoning` 属性访问。
- **格式**: `i:{"data": string}\n`
- **示例**: `i:{"data": "This reasoning has been redacted for security purposes."}\n`

##### 推理签名部分 (Reasoning Signature Part)
包含与推理相关的签名，通过 `reasoning_signature` 属性访问。
- **格式**: `j:{"signature": string}\n`
- **示例**: `j:{"signature": "abc123xyz"}\n`

##### 源部分 (Source Part)
源内容会在接收时追加到消息中，通过 `source` 属性访问。
- **格式**: `h:Source\n`
- **示例**: `h:{"sourceType":"url","id":"source-id","url":"https://example.com","title":"Example"}\n`

##### 文件部分 (File Part)
包含编码为 base64 字符串的二进制数据及其 MIME 类型，通过 `file` 属性访问。
- **格式**: `k:{data:string; mimeType:string}\n`
- **示例**: `k:{"data":"base64EncodedData","mimeType":"image/png"}\n`

##### 数据部分 (Data Part)
数据部分会被解析为 JSON 并追加到数据数组中，通过 `data` 属性访问。
- **格式**: `2:Array<JSONValue>\n`
- **示例**: `2:[{"key":"object1"},{"anotherKey":"object2"}]\n`

##### 消息注释部分 (Message Annotation Part)
消息注释会在接收时追加到消息中，通过 `annotations` 属性访问。
- **格式**: `8:Array<JSONValue>\n`
- **示例**: `8:[{"id":"message-123","other":"annotation"}]\n`

##### 错误部分 (Error Part)
错误内容会在接收时追加到消息中。
- **格式**: `3:string\n`
- **示例**: `3:"error message"\n`

##### 工具调用流开始部分 (Tool Call Streaming Start Part)
表示流式工具调用的开始，需要在该工具调用的任何增量之前发送。
- **格式**: `b:{toolCallId:string; toolName:string}\n`
- **示例**: `b:{"toolCallId":"call-456","toolName":"streaming-tool"}\n`

##### 工具调用增量部分 (Tool Call Delta Part)
表示流式工具调用的增量更新。
- **格式**: `c:{toolCallId:string; argsTextDelta:string}\n`
- **示例**: `c:{"toolCallId":"call-456","argsTextDelta":"partial arg"}\n`

##### 工具调用部分 (Tool Call Part)
表示完整的工具调用。
- **格式**: `9:{toolCallId:string; toolName:string; args:object}\n`
- **示例**: `9:{"toolCallId":"call-456","toolName":"my-tool","args":{"param":"value"}}\n`

##### 工具结果部分 (Tool Result Part)
表示工具调用的结果。
- **格式**: `a:{toolCallId:string; result:JSONValue}\n`
- **示例**: `a:{"toolCallId":"call-456","result":"Tool execution result"}\n`

##### Step 开始部分 (Start Step Part)
表示一个步骤（即后端的一次 LLM API 调用）的开始。
- **格式**: `f:{messageId?:string}\n`
- **示例**: `f:{"messageId":"step-123"}\n`

##### Step 结束部分 (Finish Step Part)
表示一个步骤的完成，包含该步骤的使用情况和是否继续的标志。
- **格式**: `e:{finishReason:string; usage:object; isContinued:boolean}\n`
- **示例**: `e:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":20},"isContinued":false}\n`

##### 完成部分 (Completion Part)
表示消息的完成，包含额外的元数据如完成原因和使用情况，必须是流中的最后一部分。
- **格式**: `d:{finishReason:string; usage:object}\n`
- **示例**: `d:{"finishReason":"stop","usage":{"promptTokens":10,"completionTokens":20}}\n`

## AI SDK v5 协议完整规范

### Server-Sent Events (SSE) 协议

AI SDK v5 使用标准的 Server-Sent Events 协议进行改进的标准化、通过 ping 保持连接、重连能力和更好的缓存处理。

#### 响应头要求
- **必需头**: `x-vercel-ai-ui-message-stream: v1`
- **内容类型**: `text/event-stream`
- **缓存控制**: `Cache-Control: no-cache`
- **连接**: `Connection: keep-alive`

#### 消息架构
AI SDK v5 引入了完全重新设计的消息系统：
- **UIMessage**: 表示界面的完整对话历史，保留所有消息部分（文本、图像、数据）、元数据（创建时间戳、生成时间）和 UI 状态
- **ModelMessage**: 针对发送给语言模型进行优化，考虑令牌输入约束，去除 UI 特定的元数据和无关内容

#### 支持的流部分类型

##### 消息开始部分 (Message Start Part)
表示新消息的开始，包含元数据。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"start","messageId":"..."}\n\n`

##### 文本部分 (Text Parts)
文本内容使用带有唯一 ID 的 start/delta/end 模式进行流式传输。

###### 文本开始部分 (Text Start Part)
表示文本块的开始。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"text-start","id":"msg_68679a454370819ca74c8eb3d04379630dd1afb72306ca5d"}\n\n`

###### 文本增量部分 (Text Delta Part)
包含文本块的增量内容。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"text-delta","id":"msg_68679a454370819ca74c8eb3d04379630dd1afb72306ca5d","delta":"Hello"}\n\n`

###### 文本结束部分 (Text End Part)
表示文本块的完成。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"text-end","id":"msg_68679a454370819ca74c8eb3d04379630dd1afb72306ca5d"}\n\n`

##### 推理部分 (Reasoning Parts)
推理内容使用带有唯一 ID 的 start/delta/end 模式进行流式传输。

###### 推理开始部分 (Reasoning Start Part)
表示推理块的开始。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"reasoning-start","id":"reasoning_123"}\n\n`

###### 推理增量部分 (Reasoning Delta Part)
包含推理块的增量内容。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"reasoning-delta","id":"reasoning_123","delta":"This is some reasoning"}\n\n`

###### 推理结束部分 (Reasoning End Part)
表示推理块的完成。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"reasoning-end","id":"reasoning_123"}\n\n`

##### 工具调用部分 (Tool Call Parts)
工具调用使用带有唯一 ID 的 start/delta/end 模式进行流式传输。

###### 工具输入开始部分 (Tool Input Start Part)
表示工具输入的开始。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"tool-input-start","id":"tool_123","toolName":"my-tool"}\n\n`

###### 工具输入增量部分 (Tool Input Delta Part)
包含工具输入参数的增量内容。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"tool-input-delta","id":"tool_123","inputTextDelta":"partial args"}\n\n`

###### 工具输入可用部分 (Tool Input Available Part)
表示工具输入参数已完整可用。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"tool-input-available","id":"tool_123","args":{"param":"value"}}\n\n`

###### 工具输出可用部分 (Tool Output Available Part)
表示工具执行结果可用。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"tool-output-available","id":"tool_123","output":"Tool execution result"}\n\n`

###### 工具输出错误部分 (Tool Output Error Part)
表示工具执行出现错误。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"tool-output-error","id":"tool_123","error":"Tool execution failed"}\n\n`

##### 数据部分 (Data Part)
包含结构化数据，会被解析为 JSON 并追加到数据数组中。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"data","id":"data_123","data":[{"key":"value"}]}\n\n`

##### 文件部分 (File Part)
包含编码为 base64 字符串的二进制数据及其 MIME 类型。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"file","id":"file_123","data":"base64EncodedData","mimeType":"image/png"}\n\n`

##### 源部分 (Source Part)
包含源引用信息，如 URL 和标题。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"source","id":"source_123","url":"https://example.com","title":"Example Source"}\n\n`

##### 错误部分 (Error Part)
表示处理过程中发生的错误。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"error","id":"error_123","error":"An error occurred"}\n\n`

##### 消息结束部分 (Message End Part)
表示消息的完成。
- **格式**: Server-Sent Event with JSON object
- **示例**: `data: {"type":"finish","id":"message_123","finishReason":"stop","usage":{"promptTokens":10,"completionTokens":20}}\n\n`

##### 流结束标记
表示整个流的结束。
- **格式**: `data: [DONE]\n\n`

### 2. 文本处理方式差异

#### AI SDK v4 文本处理
基于搜索结果确认，AI SDK v4 对文本的处理方式为：
- **单一类型**: 只有 `0:"text_content"\n` 一种文本类型
- **连续性处理**: 连续的 `0:` 类型会被追加形成完整文本
- **边界识别**: 当出现非 `0:` 类型（如 `f:` step-start 或 `e:` step-end）时，表示文本段结束
- **验证结果**: 用户猜想 2.1 是正确的 - "靠着第一个0：出现自动作为开始，连续的0:就是text-delta，不再输出0: 输出别的比如(e:或b:)就是结束"

#### AI SDK v5 文本处理
- **三阶段处理**: `text-start` → `text-delta` → `text-end`
- **明确边界**: 每个文本块都有明确的开始和结束标记
- **ID 关联**: 通过 `id` 字段关联同一文本块的不同阶段

### 4. 响应头差异

| 协议版本 | 响应头 | 值 |
|---------|--------|----|
| AI SDK v4 | `x-vercel-ai-data-stream` | `v1` |
| AI SDK v5 | `x-vercel-ai-ui-message-stream` | `v1` |

## 设计方案：双协议支持架构

### 1. 策略模式 (Strategy Pattern)

我们采用策略模式来处理两种不同的协议格式，通过配置选择使用哪种协议版本。

#### 核心组件设计

```python
# 协议策略接口
class ProtocolStrategy(ABC):
    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """获取协议特定的响应头"""
        pass
    
    @abstractmethod
    def format_chunk(self, chunk: UIMessageChunk) -> str:
        """格式化消息块为协议特定格式"""
        pass
    
    @abstractmethod
    def convert_text_sequence(self, text_chunks: List[str]) -> List[UIMessageChunk]:
        """将文本序列转换为协议特定的消息块"""
        pass

# AI SDK v4 策略实现
class AISDKv4Strategy(ProtocolStrategy):
    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "text/plain; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "x-vercel-ai-data-stream": "v1"
        }
    
    def format_chunk(self, chunk: UIMessageChunk) -> str:
        # 将 v5 格式转换为 v4 格式
        if chunk.get("type") == "text-delta":
            return f'0:"{chunk.get("delta", "")}"\n'
        elif chunk.get("type") == "start-step":
            return f'f:{{"messageId":"{chunk.get("messageId", "")}"}}\n'
        elif chunk.get("type") == "finish-step":
            return f'e:{{"finishReason":"stop","usage":{{}},"isContinued":false}}\n'
        # ... 其他类型转换
        return ""
    
    def convert_text_sequence(self, text_chunks: List[str]) -> List[UIMessageChunk]:
        # v4 只需要连续的 text-delta，不需要 start/end
        return [{
            "type": "text-delta",
            "id": str(uuid.uuid4()),
            "delta": chunk
        } for chunk in text_chunks]

# AI SDK v5 策略实现
class AISDKv5Strategy(ProtocolStrategy):
    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "x-vercel-ai-ui-message-stream": "v1"
        }
    
    def format_chunk(self, chunk: UIMessageChunk) -> str:
        # 保持 v5 SSE 格式
        json_str = json.dumps(chunk, ensure_ascii=False)
        return f"data: {json_str}\n\n"
    
    def convert_text_sequence(self, text_chunks: List[str]) -> List[UIMessageChunk]:
        # v5 需要完整的 start/delta/end 序列
        text_id = str(uuid.uuid4())
        chunks = [{"type": "text-start", "id": text_id}]
        
        for chunk in text_chunks:
            chunks.append({
                "type": "text-delta",
                "id": text_id,
                "delta": chunk
            })
        
        chunks.append({"type": "text-end", "id": text_id})
        return chunks
```

### 2. 配置管理

```python
class ProtocolConfig:
    def __init__(self, version: str = "v4"):
        self.version = version
        self.strategy = self._create_strategy()
    
    def _create_strategy(self) -> ProtocolStrategy:
        if self.version == "v4":
            return AISDKv4Strategy()
        elif self.version == "v5":
            return AISDKv5Strategy()
        else:
            raise ValueError(f"Unsupported protocol version: {self.version}")
```

### 3. 修改现有 DataStreamResponse

```python
class DataStreamResponse(StreamingResponse):
    def __init__(
        self,
        stream: AsyncGenerator[UIMessageChunk, None],
        protocol_version: str = "v4",  # 默认使用 v4
        headers: Optional[Dict[str, str]] = None,
        status: int = 200
    ):
        # 创建协议配置
        self.protocol_config = ProtocolConfig(protocol_version)
        
        # 转换流格式
        text_stream = self._convert_to_protocol_stream(stream)
        
        # 获取协议特定的响应头
        protocol_headers = self.protocol_config.strategy.get_headers()
        if headers:
            protocol_headers.update(headers)
        
        super().__init__(
            content=text_stream,
            headers=protocol_headers,
            status_code=status
        )
    
    async def _convert_to_protocol_stream(
        self, 
        data_stream: AsyncGenerator[UIMessageChunk, None]
    ) -> AsyncGenerator[str, None]:
        """根据协议版本转换流格式"""
        async for chunk in data_stream:
            formatted_chunk = self.protocol_config.strategy.format_chunk(chunk)
            if formatted_chunk:
                yield formatted_chunk
        
        # 发送终止标记
        if self.protocol_config.version == "v4":
            yield "d:{\"finishReason\":\"stop\",\"usage\":{}}\n"
        else:  # v5
            yield "data: [DONE]\n\n"
```

### 4. 文本处理适配器

```python
class TextProcessingAdapter:
    """文本处理适配器，处理 v4 和 v5 的文本差异"""
    
    def __init__(self, protocol_version: str):
        self.protocol_version = protocol_version
        self.current_text_id = None
        self.text_buffer = []
    
    def process_text_chunk(self, text: str) -> List[UIMessageChunk]:
        """处理文本块，返回协议特定的消息块"""
        if self.protocol_version == "v4":
            # v4: 直接返回 text-delta
            return [{
                "type": "text-delta",
                "id": "text-v4",  # v4 不需要真实 ID
                "delta": text
            }]
        else:
            # v5: 需要管理 start/delta/end 序列
            chunks = []
            
            if self.current_text_id is None:
                # 开始新的文本序列
                self.current_text_id = str(uuid.uuid4())
                chunks.append({
                    "type": "text-start",
                    "id": self.current_text_id
                })
            
            # 添加文本增量
            chunks.append({
                "type": "text-delta",
                "id": self.current_text_id,
                "delta": text
            })
            
            return chunks
    
    def finish_text_sequence(self) -> List[UIMessageChunk]:
        """完成文本序列"""
        if self.protocol_version == "v5" and self.current_text_id:
            chunk = {
                "type": "text-end",
                "id": self.current_text_id
            }
            self.current_text_id = None
            return [chunk]
        return []
```

## 实施计划

### 阶段 1: 核心架构实现
1. 创建 `ProtocolStrategy` 接口和具体实现
2. 修改 `DataStreamResponse` 支持协议版本参数
3. 实现 `ProtocolConfig` 配置管理

### 阶段 2: 文本处理适配
1. 实现 `TextProcessingAdapter`
2. 修改现有的文本生成逻辑
3. 确保 v4 和 v5 文本处理的正确性

### 阶段 3: 测试和验证
1. 创建 v4 和 v5 协议的测试用例
2. 验证与前端 AI SDK 的兼容性
3. 性能测试和优化

### 阶段 4: 文档和迁移
1. 更新 API 文档
2. 提供迁移指南
3. 向后兼容性保证

## 配置示例

### 使用 v4 协议（默认）
```python
response = DataStreamResponse(
    stream=your_stream,
    protocol_version="v4"  # 默认值
)
```

### 使用 v5 协议
```python
response = DataStreamResponse(
    stream=your_stream,
    protocol_version="v5"
)
```

### 环境变量配置
```bash
# 设置默认协议版本
AI_SDK_PROTOCOL_VERSION=v4
```

## 总结

通过采用策略模式和适配器模式，我们可以优雅地支持 AI SDK v4 和 v5 两种协议版本。这种设计具有以下优势：

1. **向后兼容**: 默认使用 v4 协议，确保现有应用不受影响
2. **灵活切换**: 通过配置参数轻松切换协议版本
3. **易于扩展**: 未来支持新协议版本时只需添加新的策略实现
4. **清晰分离**: 协议特定逻辑与业务逻辑分离，便于维护
5. **测试友好**: 每种协议可以独立测试和验证

这个设计方案解决了用户提出的所有关键问题，包括协议格式差异、文本处理方式差异和响应头差异，为项目提供了一个稳定且可扩展的双协议支持解决方案。