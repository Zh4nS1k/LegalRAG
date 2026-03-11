import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Avatar,
  CircularProgress,
  Alert,
  Tooltip,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import ClearIcon from '@mui/icons-material/Clear';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import InfoIcon from '@mui/icons-material/Info';
import { styled } from '@mui/material/styles';
import welcomeVideo from '../images/welcome_video_gif_legally.mp4';

const OuterContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'center',
  alignItems: 'center',
  minHeight: '100vh',
  width: '100%',
  padding: '20px',
  backgroundColor: '#f5f5f7',
  backgroundImage: 'radial-gradient(circle at top left, rgba(230,0,0,0.02), transparent 40%), radial-gradient(circle at bottom right, rgba(0,0,0,0.02), transparent 60%)',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
}));

const ChatContainer = styled(Box)(({ theme }) => ({
  width: '95%',
  maxWidth: '1000px',
  height: '90vh',
  backgroundColor: '#ffffff',
  borderRadius: '24px',
  boxShadow: '0 8px 32px rgba(0,0,0,0.06), 0 2px 8px rgba(0,0,0,0.04)',
  border: '1px solid rgba(0,0,0,0.05)',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
}));

const Header = styled(Box)(({ theme }) => ({
  background: 'rgba(255,255,255,0.85)',
  backdropFilter: 'blur(12px)',
  color: '#000000',
  padding: '20px',
  textAlign: 'center',
  position: 'relative',
  borderBottom: '1px solid rgba(0,0,0,0.06)',
  zIndex: 10,
  '& h1': {
    fontWeight: '600',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    letterSpacing: '-0.5px'
  }
}));

const StatsBadge = styled(Box)(({ theme }) => ({
  position: 'absolute',
  top: '50%',
  right: '20px',
  transform: 'translateY(-50%)',
  background: 'rgba(0,0,0,0.04)',
  color: '#000000',
  padding: '6px 14px',
  borderRadius: '20px',
  fontSize: '12px',
  fontWeight: '600',
  display: 'flex',
  alignItems: 'center',
  gap: '6px',
}));

const ChatMessages = styled(Box)(({ theme }) => ({
  flex: 1,
  padding: '24px 20px',
  overflowY: 'auto',
  backgroundColor: '#f5f5f7',
  display: 'flex',
  flexDirection: 'column',
  gap: '12px',
}));

const Message = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'isUser',
})(({ theme, isUser }) => ({
  display: 'flex',
  alignItems: 'flex-end',
  justifyContent: isUser ? 'flex-end' : 'flex-start',
  width: '100%',
  marginBottom: '8px',
}));

const MessageAvatar = styled(Avatar, {
  shouldForwardProp: (prop) => prop !== 'isUser',
})(({ theme, isUser }) => ({
  width: '34px',
  height: '34px',
  margin: '0 10px',
  backgroundColor: isUser ? '#E60000' : '#000000',
  color: 'white',
  boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
  fontSize: '16px',
}));

const MessageContent = styled(Box, {
  shouldForwardProp: (prop) => prop !== 'isUser',
})(({ theme, isUser }) => ({
  maxWidth: '75%',
  padding: '14px 20px',
  borderRadius: '24px',
  position: 'relative',
  wordWrap: 'break-word',
  backgroundColor: isUser ? '#E60000' : 'rgba(200, 200, 200, 0.15)',
  color: isUser ? '#FFFFFF' : '#000000',
  border: isUser ? 'none' : '1px solid rgba(0,0,0,0.06)',
  boxShadow: '0 2px 12px rgba(0,0,0,0.03), 0 1px 4px rgba(0,0,0,0.02)',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  lineHeight: isUser ? '1.4' : '1.6',
  fontWeight: '400',
  fontSize: '15px',
  animation: 'slideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
  '@keyframes slideUp': {
    '0%': { transform: 'translateY(10px) scale(0.98)', opacity: 0 },
    '100%': { transform: 'translateY(0) scale(1)', opacity: 1 }
  }
}));

const ModeIndicator = styled(Box)(({ theme, mode }) => ({
  position: 'absolute',
  top: '-12px',
  left: '20px',
  background: '#ffffff',
  color: mode === 'legal_rag' ? '#E60000' : '#000000',
  border: '1px solid rgba(0,0,0,0.06)',
  boxShadow: '0 2px 6px rgba(0,0,0,0.04)',
  padding: '2px 8px',
  borderRadius: '12px',
  fontSize: '10px',
  fontWeight: '600',
  letterSpacing: '0.5px'
}));

const SourcesBox = styled(Box)(({ theme }) => ({
  marginTop: '14px',
  padding: '14px',
  background: 'rgba(245,245,247,0.8)',
  borderRadius: '16px',
  border: '1px solid rgba(0,0,0,0.04)',
  fontSize: '13px',
}));

const SourceItem = styled(Box)(({ theme }) => ({
  background: 'white',
  padding: '12px',
  margin: '8px 0',
  borderRadius: '12px',
  borderLeft: '4px solid #E60000',
  boxShadow: '0 1px 4px rgba(0,0,0,0.02)',
}));

const ChatInput = styled(Box)(({ theme }) => ({
  padding: '20px 24px',
  backgroundColor: '#ffffff',
  borderTop: '1px solid rgba(0,0,0,0.06)',
  display: 'flex',
  flexDirection: 'column',
  gap: '12px',
}));

const InputContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  gap: '12px',
  alignItems: 'center',
  backgroundColor: 'rgba(0,0,0,0.03)',
  borderRadius: '50px',
  padding: '8px 8px 8px 20px',
  border: '1px solid rgba(0,0,0,0.06)',
  boxShadow: 'inset 0 1px 3px rgba(0,0,0,0.02)',
}));

const InputField = styled(TextField)(({ theme }) => ({
  flex: 1,
  '& .MuiOutlinedInput-root': {
    borderRadius: '50px',
    padding: '8px 0',
    '& fieldset': { border: 'none' },
  },
  '& textarea': {
    resize: 'none',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    fontSize: '15px',
    lineHeight: '1.4',
    padding: '0',
    color: '#000000',
  },
}));

const SendButton = styled(Button)(({ theme }) => ({
  background: '#E60000',
  color: 'white',
  borderRadius: '50px',
  padding: '10px 20px',
  fontWeight: '600',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  textTransform: 'none',
  boxShadow: '0 2px 8px rgba(230,0,0,0.2)',
  '&:hover': {
    transform: 'translateY(-1px)',
    background: '#CC0000',
    boxShadow: '0 4px 12px rgba(230,0,0,0.3)'
  },
  '&:disabled': {
    opacity: 0.5,
  },
}));

const Controls = styled(Box)(({ theme }) => ({
  display: 'flex',
  justifyContent: 'center',
  gap: '12px',
}));

const ControlButton = styled(Button)(({ theme }) => ({
  background: 'rgba(0,0,0,0.04)',
  color: '#000000',
  borderRadius: '20px',
  padding: '6px 16px',
  fontSize: '13px',
  fontWeight: '500',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  textTransform: 'none',
  boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
  '&:hover': {
    background: 'rgba(0,0,0,0.08)',
  },
}));

const TypingIndicator = styled(Box)(({ theme }) => ({
  padding: '14px 20px',
  background: '#ffffff',
  borderRadius: '24px',
  border: '1px solid rgba(0,0,0,0.06)',
  display: 'flex',
  alignItems: 'center',
  gap: '10px',
  width: 'fit-content',
  boxShadow: '0 2px 12px rgba(0,0,0,0.03), 0 1px 4px rgba(0,0,0,0.02)',
  fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
  fontSize: '14px',
}));

const ChatSection = ({
  activeSession,
  addMessageToSession,
  onClearHistory,
  onExportChat
}) => {
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({ total_vectors: 0 });
  const messagesEndRef = useRef(null);

  const messages = useMemo(() => {
    return activeSession?.messages || [];
  }, [activeSession]);

  const isEmpty = messages.length === 0;

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Load stats on mount
  useEffect(() => {
    const fetchStats = async () => {
      try {
        const statsRes = await fetch('http://localhost:8080/api/stats');
        if (statsRes.ok) {
          const statsData = await statsRes.json();
          setStats(statsData);
        }
      } catch (err) {
        console.error('Error fetching stats:', err);
      }
    };
    fetchStats();
  }, []);

  const handleInputChange = (e) => {
    setInput(e.target.value);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const sendMessage = async () => {
    const message = input.trim();
    if (!message || isTyping || !activeSession) return;

    // Add user message to session
    const userMsg = { content: message, isUser: true };
    addMessageToSession(activeSession.id, userMsg);

    setInput('');
    setIsTyping(true);
    setError(null);

    try {
      const token = localStorage.getItem('token');
      if (!token) {
        throw new Error('Требуется авторизация. Пожалуйста, войдите в систему.');
      }

      // Prepare context window (last 15 messages)
      const contextWindow = messages
        .filter(m => m.mode !== 'system') // Filter out initial welcome if needed
        .slice(-15)
        .map(m => ({
          role: m.isUser ? 'user' : 'assistant',
          content: m.content
        }));

      const response = await fetch('http://localhost:8080/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          message,
          history: contextWindow
        }),
      });

      if (response.status === 401) {
        throw new Error('Сессия истекла. Пожалуйста, войдите снова.');
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Ошибка при обработке запроса');
      }

      const data = await response.json();

      // Add AI response to session
      const aiMsg = {
        content: data.answer,
        isUser: false,
        mode: data.mode,
        sources: data.sources || [],
      };
      addMessageToSession(activeSession.id, aiMsg);

    } catch (err) {
      setError(err.message);
      const errMsg = {
        content: 'Извините, произошла ошибка: ' + err.message,
        isUser: false,
      };
      addMessageToSession(activeSession.id, errMsg);
    } finally {
      setIsTyping(false);
    }
  };

  const showStats = () => {
    alert(`📊 Статистика системы:
• Векторов в индексе: ${stats.total_vectors}
• Модели: ${stats.models?.embedding || 'N/A'}
• Reranker: ${stats.models?.reranker || 'N/A'}`);
  };

  if (!activeSession) {
    return (
      <OuterContainer>
        <Typography variant="h6" color="text.secondary">
          Выберите чат в боковой панели или создайте новый
        </Typography>
      </OuterContainer>
    );
  }

  return (
    <OuterContainer>
      <ChatContainer>
        <Header>
          <Typography variant="h5" component="h1">
            🤖 Юридический AI-ассистент
          </Typography>
          <Typography variant="body1" sx={{ mt: 0.5, opacity: 0.9 }}>
            {activeSession.title}
          </Typography>
          <StatsBadge>
            <InfoIcon fontSize="small" />
            <span>{stats.total_vectors} документов</span>
          </StatsBadge>
        </Header>

        <ChatMessages>
          {isEmpty && (
            <Box
              sx={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                height: '100%',
                opacity: 0.8
              }}
            >
              <Box
                component="video"
                autoPlay
                loop
                muted
                playsInline
                src={welcomeVideo}
                sx={{
                  width: '100%',
                  maxWidth: 360,
                  aspectRatio: '16 / 9',
                  borderRadius: 2,
                  boxShadow: '0 8px 24px rgba(0,0,0,0.2)',
                  mb: 4
                }}
              />
              <Typography variant="h5" align="center" sx={{ color: '#000000', fontWeight: 600, maxWidth: 600 }}>
                Добро пожаловать в Legally!
              </Typography>
              <Typography align="center" color="text.secondary" sx={{ mt: 2, maxWidth: 500 }}>
                Задайте любой вопрос по законодательству Казахстана, и я помогу вам найти точные ответы.
              </Typography>
            </Box>
          )}
          {messages.map((message, index) => (
            <Message key={index} isUser={message.isUser}>
              {!message.isUser && (
                <MessageAvatar isUser={message.isUser}>🤖</MessageAvatar>
              )}
              <MessageContent isUser={message.isUser}>
                {!message.isUser && message.mode && (
                  <ModeIndicator mode={message.mode}>
                    {message.mode === 'legal_rag' ? 'RAG' : 'GPT'}
                  </ModeIndicator>
                )}
                <Typography
                  component="div"
                  dangerouslySetInnerHTML={{
                    __html: message.content.replace(/\n/g, '<br>'),
                  }}
                />

                {message.sources && message.sources.length > 0 && (
                  <SourcesBox>
                    <Typography variant="subtitle2" color="primary" sx={{ mb: 1, fontWeight: 'bold' }}>
                      📚 Источники:
                    </Typography>
                    {message.sources.map((source, i) => (
                      <SourceItem key={i}>
                        <Typography variant="body2" sx={{ fontWeight: '700', color: '#000000', mb: 0.5 }}>
                          {source.title || (typeof source === 'string' ? source : 'Источник')}
                        </Typography>
                        {source.text && (
                          <Typography
                            variant="body2"
                            sx={{
                              color: '#555',
                              backgroundColor: '#f9f9f9',
                              p: 1.5,
                              borderRadius: '8px',
                              borderLeft: '4px solid #333333',
                              fontStyle: 'italic',
                              mt: 1,
                              whiteSpace: 'pre-wrap'
                            }}
                          >
                            {source.text}
                          </Typography>
                        )}
                      </SourceItem>
                    ))}
                  </SourcesBox>
                )}
              </MessageContent>
              {message.isUser && (
                <MessageAvatar isUser={message.isUser}>👤</MessageAvatar>
              )}
            </Message>
          ))}

          {isTyping && (
            <TypingIndicator>
              <CircularProgress size={16} />
              <Typography variant="body2">AI набирает сообщение...</Typography>
            </TypingIndicator>
          )}

          <div ref={messagesEndRef} />
        </ChatMessages>

        {error && (
          <Box sx={{ px: 2 }}>
            <Alert severity="error" onClose={() => setError(null)}>
              {error}
            </Alert>
          </Box>
        )}

        <ChatInput>
          <InputContainer>
            <InputField
              multiline
              minRows={1}
              maxRows={4}
              placeholder="Введите ваш вопрос..."
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              fullWidth
            />
            <Tooltip title="Отправить">
              <span>
                <SendButton
                  onClick={sendMessage}
                  disabled={!input.trim() || isTyping}
                  endIcon={<SendIcon />}
                >
                  Отправить
                </SendButton>
              </span>
            </Tooltip>
          </InputContainer>

          <Controls>
            <ControlButton startIcon={<ClearIcon />} onClick={() => onClearHistory(activeSession.id)}>
              Очистить историю
            </ControlButton>
            <ControlButton
              startIcon={<FileDownloadIcon />}
              onClick={() => onExportChat(activeSession)}
            >
              Экспорт чата
            </ControlButton>
            <ControlButton startIcon={<InfoIcon />} onClick={showStats}>
              Статистика
            </ControlButton>
          </Controls>
        </ChatInput>
      </ChatContainer>
    </OuterContainer>
  );
};

export default ChatSection;
