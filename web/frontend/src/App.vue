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
              FastAPI + Vue.js + @ai-sdk/vue Integration Demo
            </p>
          </div>
          <div class="flex items-center gap-4">
            <!-- 模式选择 -->
            <div class="flex items-center gap-2">
              <label class="text-sm font-medium text-gray-700">Mode:</label>
              <select 
                v-model="streamMode" 
                class="input text-sm px-3 py-1 min-w-0 w-auto"
                @change="clearMessages"
              >
                <option value="auto">Auto Mode</option>
                <option value="manual">Manual Mode</option>
              </select>
            </div>
            <!-- 协议版本选择 -->
            <div class="flex items-center gap-2">
              <label class="text-sm font-medium text-gray-700">Protocol:</label>
              <select 
                v-model="protocolVersion" 
                class="input text-sm px-3 py-1 min-w-0 w-auto"
                @change="clearMessages"
              >
                <option value="v4">v4 (with usage)</option>
                <option value="v5">v5 (latest)</option>
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
                {{ backendStatus === 'connected' ? 'Connected' : 'Disconnected' }}
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>

    <!-- 主要内容区域 -->
    <main class="container mx-auto px-4 py-6">
      <div class="max-w-4xl mx-auto">
        <!-- Mode Description Card -->
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
                  Current Mode: {{ streamMode === 'auto' ? 'Auto Mode' : 'Manual Mode' }} | Protocol: {{ protocolVersion }}
                </h3>
                <p class="text-sm text-gray-600">
                  <template v-if="streamMode === 'auto'">
                    Uses <code class="bg-gray-100 px-1 rounded">to_data_stream_response</code> 
                    for automatic stream processing, suitable for simple conversation scenarios. Thread safety guaranteed through instance isolation.
                  </template>
                  <template v-else>
                    Uses LangChain callbacks for precise control over each stage of the stream, suitable for complex workflow scenarios.
                    Thread safety guaranteed through instance isolation and context managers.
                  </template>
                  <br><br>
                  <strong>Protocol {{ protocolVersion }}:</strong> 
                  <template v-if="protocolVersion === 'v4'">
                    Includes usage information (token counts) in termination markers for cost tracking.
                  </template>
                  <template v-else>
                    Latest protocol version with enhanced features and optimizations.
                  </template>
                </p>
              </div>
            </div>
          </div>
        </div>

        <!-- Chat Interface -->
        <div class="card">
          <div class="card-header">
            <div class="flex items-center justify-between">
              <h2 class="text-lg font-semibold text-gray-900">Chat Conversation</h2>
              <button 
                @click="clearMessages" 
                class="btn btn-secondary btn-sm"
                :disabled="isLoading"
              >
                Clear Chat
              </button>
            </div>
          </div>
          
          <div class="card-body">
            <!-- Message List -->
            <div 
              ref="messagesContainer"
              class="h-96 overflow-y-auto mb-4 p-4 bg-gray-50 rounded-lg"
            >
              <div v-if="messages.length === 0" class="text-center text-gray-500 mt-8">
                <svg class="w-12 h-12 mx-auto mb-4 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-3.582 8-8 8a8.959 8.959 0 01-4.906-1.471L3 21l2.529-5.094A8.959 8.959 0 013 12c0-4.418 3.582-8 8-8s8 3.582 8 8z"></path>
                </svg>
                <p>Start chatting! You can ask me anything.</p>
                <p class="text-sm mt-2">I can query internal employee birthday information.</p>
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
                  <div class="max-w-xs lg:max-w-md px-4 py-2 rounded-lg shadow-sm" :class="message.error ? 'bg-red-50 border border-red-200' : 'bg-white border'">
                    <div class="flex items-center gap-2 mb-2">
                      <div class="w-6 h-6 rounded-full flex items-center justify-center" :class="message.error ? 'bg-red-100' : 'bg-green-100'">
                        <svg v-if="message.error" class="w-3 h-3 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                        </svg>
                        <svg v-else class="w-3 h-3 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path>
                        </svg>
                      </div>
                      <span class="text-xs" :class="message.error ? 'text-red-600' : 'text-gray-500'">
                        {{ message.error ? 'Error' : 'AI Assistant' }}
                      </span>
                    </div>
                    
                    <!-- 渲染消息的各个部分 -->
                    <div class="message-content">
                      <template v-for="(part, partIndex) in (message.parts || [])" :key="partIndex">
                        <!-- 文本部分 -->
                        <div v-if="part.type === 'text'" class="whitespace-pre-wrap" :class="message.error ? 'text-red-700' : ''">
                          {{ part.text }}
                        </div>
                        
                        <!-- 工具调用部分 -->
                        <div v-else-if="part.type === 'tool-invocation'" class="tool-call-container my-2">
                          <div class="tool-call-header">
                            <div class="flex items-center gap-2">
                              <svg class="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
                              </svg>
                              <span class="tool-name">{{ part.toolInvocation?.toolName || 'Tool' }}</span>
                            </div>
                          </div>
                          <div class="tool-call-content">
                            <div class="tool-input" v-if="part.toolInvocation?.args">
                              <strong>Input:</strong> {{ JSON.stringify(part.toolInvocation.args, null, 2) }}
                            </div>
                            <div class="tool-result" v-if="part.toolInvocation?.result">
                              <strong>Result:</strong>
                              <div class="employee-info" v-if="part.toolInvocation.result && typeof part.toolInvocation.result === 'object'">
                                <div v-for="(value, key) in part.toolInvocation.result" :key="key" class="info-item">
                                  <span class="info-label">{{ formatFieldLabel(key) }}:</span>
                                  <span class="info-value" :class="getFieldClass(key, value)">{{ value }}</span>
                                </div>
                              </div>
                              <div v-else class="text-result">
                                {{ part.toolInvocation.result }}
                              </div>
                            </div>
                          </div>
                        </div>
                        
                        <!-- 步骤开始标记 -->
                        <div v-else-if="part.type === 'step-start'" class="step-divider my-2">
                          <div class="w-full h-px bg-gray-200"></div>
                        </div>
                        
                        <!-- 回退到原始内容显示 -->
                        <div v-else class="whitespace-pre-wrap" :class="message.error ? 'text-red-700' : ''" v-html="formatMessageContent(message.content)">
                        </div>
                      </template>
                    </div>
                    <div v-if="message.error && message.errorDetails" class="mt-2 p-2 bg-red-100 rounded text-xs text-red-600">
                      <strong>Error Details:</strong> {{ message.errorDetails }}
                    </div>
                  </div>
                </div>
              </div>
              
              <!-- Custom Data Display (Manual Mode) -->
              <div v-if="data && data.length > 0 && streamMode === 'manual'" class="mb-4">
                <div class="bg-purple-50 border border-purple-200 rounded-lg p-3">
                  <div class="text-xs font-semibold text-purple-700 mb-2">📊 Manual Mode Data</div>
                  <div v-for="(item, index) in data" :key="index" class="text-xs text-purple-600 mb-1">
                    <code class="bg-purple-100 px-1 rounded">{{ JSON.stringify(item) }}</code>
                  </div>
                </div>
              </div>
              
              <!-- Loading Indicator -->
              <div v-if="isLoading" class="flex justify-start">
                <div class="max-w-xs lg:max-w-md px-4 py-2 bg-white border rounded-lg shadow-sm">
                  <div class="flex items-center gap-2">
                    <div class="loading"></div>
                    <span class="text-sm text-gray-500">AI is thinking...</span>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- Tool Information Display -->
            <div class="mb-4 p-2 bg-blue-50 border border-blue-200 rounded-lg">
              <div class="flex items-center gap-2 mb-2">
                <svg class="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <span class="text-sm font-medium text-blue-800">Available Tools</span>
              </div>
              <div class="bg-white p-2 rounded border border-blue-100">
                <div class="flex items-center gap-2">
                  <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                  <span class="text-sm text-gray-700">Employee Birthday Query System</span>
                  <span class="text-xs text-gray-500 ml-auto">Internal Data</span>
                </div>
                <p class="text-xs text-gray-600 mt-1 ml-4">Query internal employee birthday information</p>
              </div>
            </div>
            
            <!-- Quick Questions -->
            <div class="mb-4">
              <div class="flex items-center gap-2 mb-2">
                <svg class="w-4 h-4 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                <span class="text-sm font-medium text-gray-700">Quick Questions</span>
              </div>
              <div class="flex flex-wrap gap-2">
                <button 
                  @click="quickQuestion('Please query Zhang San\'s birthday information and calculate the time difference with Einstein\'s birthday')"
                  class="btn btn-secondary btn-sm"
                  :disabled="isLoading"
                >
                  Compare Zhang San's Birthday with Einstein's
                </button>
                <button 
                  @click="quickQuestion('Please get Li Si\'s birthday and tell me how many days it is from Einstein\'s birthday')"
                  class="btn btn-secondary btn-sm"
                  :disabled="isLoading"
                >
                  Days Difference: Li Si vs Einstein
                </button>
                <button 
                  @click="quickQuestion('Query Wang Wu\'s birthday information')"
                  class="btn btn-secondary btn-sm"
                  :disabled="isLoading"
                >
                  Query Wang Wu's Birthday
                </button>
              </div>
            </div>
            
            <!-- Input Area -->
            <form @submit.prevent="sendMessage" class="flex gap-2">
              <input
                v-model="input"
                type="text"
                placeholder="Enter your message..."
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
                  Sending
                </template>
                <template v-else>
                  <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path>
                  </svg>
                  Send
                </template>
              </button>
            </form>
          </div>
        </div>
        
        <!-- Technical Features Card -->
        <div class="card mt-6">
          <div class="card-body">
            <h3 class="font-semibold text-gray-900 mb-3">Technical Features</h3>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                <span>Thread Safety (Instance Isolation)</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-blue-500 rounded-full"></div>
                <span>Streaming Response</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-purple-500 rounded-full"></div>
                <span>LangChain Agent</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-orange-500 rounded-full"></div>
                <span>@ai-sdk/vue Integration</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-red-500 rounded-full"></div>
                <span>Tool Calling (Employee Birthday Query)</span>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 bg-teal-500 rounded-full"></div>
                <span>FastAPI Backend</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { useChat } from '@ai-sdk/vue'

// 响应式数据
const streamMode = ref('auto')
const protocolVersion = ref('v4')
const backendStatus = ref('disconnected')
const messagesContainer = ref(null)

// 计算API端点
const apiEndpoint = computed(() => {
  return streamMode.value === 'auto' ? '/api/chat/auto' : '/api/chat/manual'
})

// 使用 @ai-sdk/vue 的 useChat hook
const {
  messages,
  input,
  handleSubmit,
  isLoading,
  setMessages,
  data
} = useChat({
  api: apiEndpoint.value,
  streamProtocol: 'data', // 使用 AI SDK Data Stream Protocol
  body: {
    message_id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    stream_mode: streamMode.value,
    protocol_version: protocolVersion.value
  },
  onError: (error) => {
    console.error('Chat error:', error)
    
    // 添加错误消息到聊天记录
    const errorMessage = {
      id: `error_${Date.now()}`,
      role: 'assistant',
      content: 'Sorry, an error occurred while processing your request.',
      error: true,
      errorDetails: error.message || 'Unknown error'
    }
    
    // 将错误消息添加到现有消息中
    setMessages([...messages.value, errorMessage])
  },
  onFinish: (message) => {
    console.log('Chat finished:', message)
    console.log('Current mode:', streamMode.value)
    console.log('Custom data received:', data.value)
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

// 格式化消息内容，用于回退显示
const formatMessageContent = (content) => {
  if (!content) return ''
  return content
}

// 格式化字段标签
const formatFieldLabel = (key) => {
  const labelMap = {
    'employee': 'Employee',
    'department': 'Department', 
    'birthday': 'Birthday',
    'access_level': 'Access Level',
    'name': 'Name',
    'id': 'ID'
  }
  return labelMap[key] || key.charAt(0).toUpperCase() + key.slice(1)
}

// 获取字段样式类
const getFieldClass = (key, value) => {
  const classMap = {
    'employee': 'employee-field',
    'name': 'name-highlight',
    'department': 'department-field',
    'birthday': 'birthday-field',
    'access_level': 'access-field'
  }
  return classMap[key] || ''
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

.bg-red-50 {
  background-color: #fef2f2;
}

.bg-red-100 {
  background-color: #fee2e2;
}

.border-red-200 {
  border-color: #fecaca;
}

.text-red-600 {
  color: #dc2626;
}

.text-red-700 {
  color: #b91c1c;
}

/* 结构化数据高亮样式 */
.employee-field {
  background: linear-gradient(120deg, #e0f2fe 0%, #b3e5fc 100%);
  color: #01579b;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 600;
  border-left: 3px solid #0288d1;
  display: inline-block;
  margin: 1px 0;
}

.department-field {
  background: linear-gradient(120deg, #f3e5f5 0%, #e1bee7 100%);
  color: #4a148c;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 600;
  border-left: 3px solid #7b1fa2;
  display: inline-block;
  margin: 1px 0;
}

.birthday-field {
  background: linear-gradient(120deg, #fff3e0 0%, #ffcc80 100%);
  color: #e65100;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 600;
  border-left: 3px solid #f57c00;
  display: inline-block;
  margin: 1px 0;
}

.access-field {
  background: linear-gradient(120deg, #e8f5e8 0%, #c8e6c9 100%);
  color: #1b5e20;
  padding: 2px 6px;
  border-radius: 4px;
  font-weight: 600;
  border-left: 3px solid #388e3c;
  display: inline-block;
  margin: 1px 0;
}

.name-highlight {
  background: linear-gradient(120deg, #ddd6fe 0%, #c4b5fd 100%);
  color: #6b21a8;
  padding: 2px 4px;
  border-radius: 3px;
  font-weight: 500;
  border: 1px solid #8b5cf6;
}

.famous-name-highlight {
  background: linear-gradient(120deg, #fef3c7 0%, #fde68a 100%);
  color: #92400e;
  padding: 2px 4px;
  border-radius: 3px;
  font-weight: 500;
  border: 1px solid #f59e0b;
}

.date-highlight {
  background: linear-gradient(120deg, #fce4ec 0%, #f8bbd9 100%);
  color: #880e4f;
  padding: 2px 4px;
  border-radius: 3px;
  font-weight: 500;
  border: 1px solid #ad1457;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
}

.department-highlight {
  background: linear-gradient(120deg, #e0f7fa 0%, #b2ebf2 100%);
  color: #006064;
  padding: 2px 4px;
  border-radius: 3px;
  font-weight: 500;
  border: 1px solid #00838f;
}

/* 悬停效果 */
.employee-field:hover,
.department-field:hover,
.birthday-field:hover,
.access-field:hover {
  transform: scale(1.02);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  transition: all 0.2s ease;
}

.name-highlight:hover {
  background: linear-gradient(120deg, #c4b5fd 0%, #a78bfa 100%);
  transform: scale(1.02);
  transition: all 0.2s ease;
}

.famous-name-highlight:hover {
  background: linear-gradient(120deg, #fde68a 0%, #fcd34d 100%);
  transform: scale(1.02);
  transition: all 0.2s ease;
}

.date-highlight:hover {
  background: linear-gradient(120deg, #f8bbd9 0%, #f48fb1 100%);
  transform: scale(1.02);
  transition: all 0.2s ease;
}

.department-highlight:hover {
  background: linear-gradient(120deg, #b2ebf2 0%, #80deea 100%);
  transform: scale(1.02);
  transition: all 0.2s ease;
}

/* 工具调用样式 */
.tool-call-container {
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
  border: 1px solid #0ea5e9;
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
}

.tool-call-header {
  margin-bottom: 8px;
}

.tool-name {
  font-weight: 600;
  color: #0369a1;
  font-size: 0.875rem;
}

.tool-call-content {
  background: rgba(255, 255, 255, 0.7);
  border-radius: 6px;
  padding: 8px;
  font-size: 0.875rem;
}

.tool-input {
  margin-bottom: 8px;
  color: #374151;
}

.tool-result {
  color: #374151;
}

.employee-info {
  margin-top: 8px;
}

.info-item {
  display: flex;
  margin-bottom: 4px;
  align-items: center;
  gap: 8px;
}

.info-label {
  font-weight: 600;
  color: #4b5563;
  min-width: 100px;
}

.info-value {
  color: #111827;
}

.text-result {
  margin-top: 4px;
  padding: 4px 8px;
  background: #f9fafb;
  border-radius: 4px;
  border-left: 3px solid #10b981;
}

.step-divider {
  margin: 12px 0;
}

.my-2 {
  margin-top: 0.5rem;
  margin-bottom: 0.5rem;
}

.w-full {
  width: 100%;
}

.h-px {
  height: 1px;
}

.bg-gray-200 {
  background-color: #e5e7eb;
}
</style>