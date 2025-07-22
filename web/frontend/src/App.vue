<template>
  <div id="app">
    <!-- 头部 -->
    <header class="bg-white shadow-sm border-b">
      <div class="container mx-auto px-4 py-4">
        <div class="flex items-center justify-between">
          <div>
            <h1 class="text-2xl font-bold text-gray-900">
              LangChain AI SDK Adapter Demo
            </h1>
            <p class="text-sm text-gray-600 mt-1">
              FastAPI + Vue.js + @ai-sdk/vue 集成示例
            </p>
          </div>
          <div class="flex items-center gap-4">
            <!-- 模式选择 -->
            <div class="flex items-center gap-2">
              <label class="text-sm font-medium text-gray-700">模式:</label>
              <select 
                v-model="streamMode" 
                class="input text-sm px-3 py-1 min-w-0 w-auto"
                @change="clearMessages"
              >
                <option value="auto">自动模式</option>
                <option value="manual">手动模式</option>
              </select>
            </div>
            <!-- 状态指示器 -->
            <div class="flex items-center gap-2">
              <div 
                :class="[
                  'w-2 h-2 rounded-full',
                  backendStatus === 'connected' ? 'bg-green-500' : 'bg-red-500'
                ]"
              ></div>
              <span class="text-sm text-gray-600">
                {{ backendStatus === 'connected' ? '已连接' : '未连接' }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>

    <!-- 主要内容区域 -->
    <main class="container mx-auto px-4 py-6">
      <div class="max-w-4xl mx-auto">
        <!-- 模式说明卡片 -->
        <div class="card mb-6">
          <div class="card-body">
            <div class="flex items-start gap-4">
              <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <svg class="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                  </svg>
                </div>
              </div>
              <div class="flex-1">
                <h3 class="font-semibold text-gray-900 mb-2">
                  当前模式: {{ streamMode === 'auto' ? '自动模式' : '手动模式' }}
                </h3>
                <p class="text-sm text-gray-600">
                  <template v-if="streamMode === 'auto'">
                    使用 <code class="bg-gray-100 px-1 rounded">to_data_stream_response</code> 
                    自动处理流，适合简单对话场景。线程安全通过实例隔离保证。
                  </template>
                  <template v-else>
                    使用 LangChain 回调精确控制流的每个阶段，适合复杂工作流场景。
                    线程安全通过实例隔离和上下文管理器保证。
                  </template>
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- 聊天界面 -->
        <div class="card">
          <div class="card-header">
            <div class="flex items-center justify-between">
              <h2 class="text-lg font-semibold text-gray-900">聊天对话</h2>
              <button 
                @click="clearMessages" 
                class="btn btn-secondary btn-sm"
                :disabled="isLoading"
              >
                清空对话
              </button>
            </div>
          </div>
          
          <div class="card-body">
            <!-- 消息列表 -->
            <div 
              ref="messagesContainer"
              class="h-96 overflow-y-auto mb-4 p-4 bg-gray-50 rounded-lg"
            >
              <div v-if="messages.length === 0" class="text-center text-gray-500 mt-8">
                <svg class="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.959 8.959 0 01-4.906-1.471L3 21l2.529-5.094A8.959 8.959 0 013 12c0-4.418 3.582-8 8-8s8 3.582 8 8z"></path>
                </svg>
                <p>开始对话吧！你可以问我任何问题。</p>
                <p class="text-sm mt-2">我可以查询公司内部员工生日信息。</p>
              </div>
              
              <div v-for="(message, index) in messages" :key="index" class="mb-4">
                <!-- 用户消息 -->
                <div v-if="message.role === 'user'" class="flex justify-end">
                  <div class="max-w-xs lg:max-w-md px-4 py-2 bg-blue-500 text-white rounded-lg">
                    {{ message.content }}
                  </div>
                </div>
                
                <!-- AI 消息 -->
                <div v-else class="flex justify-start">
                  <div class="max-w-xs lg:max-w-md px-4 py-2 bg-white border rounded-lg shadow-sm">
                    <div class="flex items-center gap-2 mb-2">
                      <div class="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center">
                        <svg class="w-3 h-3 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
                        </svg>
                      </div>
                      <span class="text-xs text-gray-500">AI Assistant</span>
                    </div>
                    <div class="whitespace-pre-wrap">{{ message.content }}</div>
                  </div>
                </div>
              </div>
              
              <!-- 加载指示器 -->
              <div v-if="isLoading" class="flex justify-start">
                <div class="max-w-xs lg:max-w-md px-4 py-2 bg-white border rounded-lg shadow-sm">
                  <div class="flex items-center gap-2">
                    <div class="loading"></div>
                    <span class="text-sm text-gray-500">AI 正在思考...</span>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- 工具信息显示 -->
            <div class="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div class="flex items-center gap-2 mb-2">
                <svg class="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <span class="text-sm font-medium text-blue-800">可用工具</span>
              </div>
              <div class="bg-white p-2 rounded border border-blue-100">
                <div class="flex items-center gap-2">
                  <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span class="text-sm text-gray-700">员工生日查询系统</span>
                  <span class="text-xs text-gray-500 ml-auto">内部数据</span>
                </div>
                <p class="text-xs text-gray-600 mt-1 ml-4">查询公司内部员工的生日信息</p>
              </div>
            </div>
            
            <!-- 便捷提问 -->
            <div class="mb-4">
              <div class="flex items-center gap-2 mb-2">
                <svg class="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <span class="text-sm font-medium text-gray-700">快速提问</span>
              </div>
              <div class="flex flex-wrap gap-2">
                <button 
                  @click="quickQuestion('请查询张三的生日信息，并计算与爱因斯坦生日的时间差')"
                  class="btn btn-secondary btn-sm"
                  :disabled="isLoading"
                >
                  查询张三生日与爱因斯坦生日对比
                </button>
                <button 
                  @click="quickQuestion('请获取李四的生日，并告诉我距离爱因斯坦生日有多少天')"
                  class="btn btn-secondary btn-sm"
                  :disabled="isLoading"
                >
                  李四生日与爱因斯坦生日天数差
                </button>
                <button 
                  @click="quickQuestion('查询王五的生日信息')"
                  class="btn btn-secondary btn-sm"
                  :disabled="isLoading"
                >
                  查询王五生日
                </button>
              </div>
            </div>
            
            <!-- 输入区域 -->
            <form @submit.prevent="sendMessage" class="flex gap-2">
              <input
                v-model="input"
                type="text"
                placeholder="输入你的消息..."
                class="input flex-1"
                :disabled="isLoading"
                @keydown.enter.prevent="sendMessage"
              />
              <button 
                type="submit" 
                class="btn btn-primary"
                :disabled="isLoading || !input.trim()"
              >
                <template v-if="isLoading">
                  <div class="loading"></div>
                  发送中
                </template>
                <template v-else>
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                  </svg>
                  发送
                </template>
              </button>
            </form>
          </div>
        </div>
        
        <!-- 技术信息卡片 -->
        <div class="card mt-6">
          <div class="card-body">
            <h3 class="font-semibold text-gray-900 mb-3">技术特性</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>线程安全 (实例隔离)</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>流式响应</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span>LangChain Agent</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-orange-500 rounded-full"></div>
                <span>@ai-sdk/vue 集成</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-red-500 rounded-full"></div>
                <span>工具调用 (员工生日查询)</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-teal-500 rounded-full"></div>
                <span>FastAPI 后端</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick, watch } from 'vue'
import { useChat } from '@ai-sdk/vue'

// 响应式数据
const streamMode = ref('auto')
const backendStatus = ref('disconnected')
const messagesContainer = ref(null)

// 使用 @ai-sdk/vue 的 useChat hook
const {
  messages,
  input,
  handleSubmit,
  isLoading,
  setMessages
} = useChat({
  api: () => {
    // 根据选择的模式返回不同的 API 端点
    return streamMode.value === 'auto' ? '/api/chat/auto' : '/api/chat/manual'
  },
  body: {
    stream_mode: streamMode.value
  },
  onError: (error) => {
    console.error('Chat error:', error)
    // 可以在这里添加错误处理逻辑
  },
  onFinish: (message) => {
    console.log('Chat finished:', message)
    // 滚动到底部
    scrollToBottom()
  }
})

// 方法
const sendMessage = () => {
  if (!input.value.trim() || isLoading.value) return
  
  handleSubmit()
  
  // 滚动到底部
  nextTick(() => {
    scrollToBottom()
  })
}

const clearMessages = () => {
  setMessages([])
}

const quickQuestion = (question) => {
  if (isLoading.value) return
  
  input.value = question
  sendMessage()
}

const scrollToBottom = () => {
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

const checkBackendStatus = async () => {
  try {
    const response = await fetch('/api/health')
    if (response.ok) {
      backendStatus.value = 'connected'
    } else {
      backendStatus.value = 'disconnected'
    }
  } catch (error) {
    console.error('Backend health check failed:', error)
    backendStatus.value = 'disconnected'
  }
}

// 监听消息变化，自动滚动到底部
watch(messages, () => {
  nextTick(() => {
    scrollToBottom()
  })
}, { deep: true })

// 组件挂载时检查后端状态
onMounted(() => {
  checkBackendStatus()
  
  // 定期检查后端状态
  setInterval(checkBackendStatus, 30000) // 每30秒检查一次
})
</script>

<style scoped>
/* 组件特定样式 */
.grid {
  display: grid;
}

.grid-cols-1 {
  grid-template-columns: repeat(1, minmax(0, 1fr));
}

@media (min-width: 768px) {
  .md\:grid-cols-2 {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

.max-w-xs {
  max-width: 20rem;
}

.max-w-4xl {
  max-width: 56rem;
}

@media (min-width: 1024px) {
  .lg\:max-w-md {
    max-width: 28rem;
  }
}

.whitespace-pre-wrap {
  white-space: pre-wrap;
}

.bg-gray-50 {
  background-color: #f9fafb;
}

.bg-gray-100 {
  background-color: #f3f4f6;
}

.bg-blue-100 {
  background-color: #dbeafe;
}

.bg-green-100 {
  background-color: #dcfce7;
}

.bg-blue-500 {
  background-color: #3b82f6;
}

.bg-green-500 {
  background-color: #10b981;
}

.bg-red-500 {
  background-color: #ef4444;
}

.bg-purple-500 {
  background-color: #8b5cf6;
}

.bg-orange-500 {
  background-color: #f97316;
}

.bg-teal-500 {
  background-color: #14b8a6;
}

.text-blue-600 {
  color: #2563eb;
}

.text-green-600 {
  color: #16a34a;
}

.text-gray-300 {
  color: #d1d5db;
}

.text-gray-500 {
  color: #6b7280;
}

.text-gray-600 {
  color: #4b5563;
}

.text-gray-700 {
  color: #374151;
}

.text-gray-900 {
  color: #111827;
}

.border-b {
  border-bottom-width: 1px;
}

.h-96 {
  height: 24rem;
}

.w-2 {
  width: 0.5rem;
}

.h-2 {
  height: 0.5rem;
}

.w-3 {
  width: 0.75rem;
}

.h-3 {
  height: 0.75rem;
}

.w-4 {
  width: 1rem;
}

.h-4 {
  height: 1rem;
}

.w-6 {
  width: 1.5rem;
}

.h-6 {
  height: 1.5rem;
}

.w-8 {
  width: 2rem;
}

.h-8 {
  height: 2rem;
}

.w-12 {
  width: 3rem;
}

.h-12 {
  height: 3rem;
}

.text-2xl {
  font-size: 1.5rem;
  line-height: 2rem;
}

.text-xs {
  font-size: 0.75rem;
  line-height: 1rem;
}

.mx-auto {
  margin-left: auto;
  margin-right: auto;
}

.mb-2 {
  margin-bottom: 0.5rem;
}

.mb-3 {
  margin-bottom: 0.75rem;
}

.mb-4 {
  margin-bottom: 1rem;
}

.mb-6 {
  margin-bottom: 1.5rem;
}

.mt-1 {
  margin-top: 0.25rem;
}

.mt-2 {
  margin-top: 0.5rem;
}

.mt-6 {
  margin-top: 1.5rem;
}

.mt-8 {
  margin-top: 2rem;
}

.px-1 {
  padding-left: 0.25rem;
  padding-right: 0.25rem;
}

.px-3 {
  padding-left: 0.75rem;
  padding-right: 0.75rem;
}

.py-1 {
  padding-top: 0.25rem;
  padding-bottom: 0.25rem;
}

.py-6 {
  padding-top: 1.5rem;
  padding-bottom: 1.5rem;
}

.min-w-0 {
  min-width: 0px;
}

.w-auto {
  width: auto;
}

.flex-shrink-0 {
  flex-shrink: 0;
}

.flex-1 {
  flex: 1 1 0%;
}

.justify-end {
  justify-content: flex-end;
}

.justify-start {
  justify-content: flex-start;
}

.overflow-y-auto {
  overflow-y: auto;
}

code {
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
}

.bg-blue-50 {
  background-color: #eff6ff;
}

.border-blue-100 {
  border-color: #dbeafe;
}

.border-blue-200 {
  border-color: #bfdbfe;
}

.text-blue-800 {
  color: #1e40af;
}

.ml-auto {
  margin-left: auto;
}

.ml-4 {
  margin-left: 1rem;
}

.flex-wrap {
  flex-wrap: wrap;
}
</style>