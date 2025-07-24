import React, { useEffect, useRef } from 'react';
import { useChat } from '@ai-sdk/react';
import './App.css';

interface BackendStatus {
  status: 'connected' | 'disconnected' | 'checking';
}

function App() {
  const { messages, input, handleInputChange, handleSubmit, status, error, setMessages } = useChat();
  const isLoading = status === 'streaming';
  
  const [backendStatus, setBackendStatus] = React.useState<BackendStatus['status']>('checking');
  const messagesContainerRef = useRef<HTMLDivElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  };

  const checkBackendStatus = async () => {
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        setBackendStatus('connected');
      } else {
        setBackendStatus('disconnected');
      }
    } catch (error) {
      console.error('Backend health check failed:', error);
      setBackendStatus('disconnected');
    }
  };

  // Custom function to send quick questions
  const sendQuickMessage = (content: string) => {
    if (isLoading) return;
    
    // Create a synthetic form event to trigger handleSubmit
    const syntheticEvent = {
      preventDefault: () => {},
      target: { elements: { prompt: { value: content } } }
    } as any;
    
    handleSubmit(syntheticEvent);
  };
  
  const quickQuestion = (question: string) => {
    if (isLoading) return;
    sendQuickMessage(question);
  };

  const clearMessages = () => {
    setMessages([]);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    checkBackendStatus();
    const interval = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const quickQuestions = [
    "What is LangChain?",
    "How to use AI SDK?",
    "Explain streaming response",
    "What is RAG?"
  ];

  return (
    <div className="App">
      <div className="container">
        {/* Header */}
        <div className="header">
          <h1 className="title">AI Chat Assistant</h1>
          <p className="subtitle">Intelligent conversation with AI assistant</p>
          <div className={`status ${
            backendStatus === 'checking' ? 'checking' :
            backendStatus === 'connected' ? 'connected' :
            'error'
          }`}>
            <div className="status-dot"></div>
            <span>
              {backendStatus === 'checking' ? 'Checking backend connection...' :
               backendStatus === 'connected' ? 'Backend connected' :
               'Backend connection failed'}
            </span>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-content">
          {/* Chat Container */}
          <div className="chat-container">
            {/* Chat Header */}
            <div className="chat-header">
              <h2 className="chat-title">Chat</h2>
              <button
                onClick={clearMessages}
                className="clear-btn"
              >
                Clear Chat
              </button>
            </div>

            {/* Messages */}
            <div 
               ref={messagesEndRef}
               className="messages-container"
             >
              {messages.length === 0 ? (
                <div style={{textAlign: 'center', color: '#6b7280', marginTop: '2rem'}}>
                  <p style={{fontSize: '1.125rem', marginBottom: '0.5rem'}}>👋 Welcome to AI Assistant!</p>
                  <p>Start a conversation or select quick questions on the right</p>
                </div>
              ) : (
                messages.map((message, index) => (
                  <div
                    key={index}
                    className={`message ${message.role}`}
                  >
                    <div className="message-avatar">
                      {message.role === 'user' ? 'U' : 'AI'}
                    </div>
                    <div className="message-content">
                      {message.parts?.map((part: any, index: number) => {
                        if (part.type === 'text') {
                          return <span key={index}>{part.text}</span>;
                        }
                        return null;
                      }) || ''}
                    </div>
                  </div>
                ))
              )}
              
              {/* Loading indicator */}
              {isLoading && (
                <div className="loading-indicator">
                  <div className="loading-dots">
                    <div className="loading-dot"></div>
                    <div className="loading-dot"></div>
                    <div className="loading-dot"></div>
                  </div>
                  <span>AI is thinking...</span>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="input-container">
              <form onSubmit={handleSubmit} className="input-form">
                <input
                  name="prompt"
                  type="text"
                  value={input}
                  onChange={handleInputChange}
                  placeholder="Enter your question..."
                  className="input-field"
                  disabled={isLoading}
                />
                <button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  className="send-btn"
                >
                  {isLoading ? 'Sending...' : 'Send'}
                </button>
              </form>
            </div>
          </div>

          {/* Sidebar */}
          <div className="sidebar">
            <h3 className="sidebar-title">Quick Questions</h3>
            <div className="quick-questions">
              {quickQuestions.map((question, index) => (
                <button
                  key={index}
                  onClick={() => quickQuestion(question)}
                  disabled={isLoading}
                  className="quick-question-btn"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error-message">
            Error: {error.message}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
