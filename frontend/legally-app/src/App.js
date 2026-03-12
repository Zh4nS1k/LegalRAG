import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
} from 'react-router-dom';
import Header from './components/Header';
import UploadSection from './components/UploadSection';
import LoadingSection from './components/LoadingSection';
import ResultSection from './components/ResultSection';
import HistorySection from './components/HistorySection';
import Footer from './components/Footer';
import AuthPage from './components/AuthPage';
import AuthCallback from './components/AuthCallback';
import ProtectedRoute from './components/ProtectedRoute';
import AdminRoute from './components/AdminRoute';
import ReviewerRoute from './components/ReviewerRoute';
import ProfileSection from './components/ProfileSection';
import ChatSection from './components/ChatSection';
import EvaluationAdmin from './components/Admin/EvaluationAdmin';
import ReviewerDashboard from './components/Reviewer/ReviewerDashboard';
import EvaluationForm from './components/Reviewer/EvaluationForm';
import { CContainer } from '@coreui/react';
import {
  Home as HomeIcon,
  History as HistoryIcon,
  Person as ProfileIcon,
  Chat as ChatIcon,
  VerifiedUser as EvalIcon,
} from '@mui/icons-material';
import { Box, Typography, Button, Modal, Fade, Avatar } from '@mui/material';
import Sidebar from './components/Sidebar';
import { useChatHistory } from './hooks/useChatHistory';
import './styles/index.css';
import './styles/hitl.css';

function AppWrapper() {
  return (
    <Router>
      <App />
    </Router>
  );
}

function App() {
  const navigate = useNavigate();
  const abortControllerRef = useRef(null);
  const chatHistory = useChatHistory();
  const [appState, setAppState] = useState({
    isLoading: false,
    fileInfo: null,
    analysisData: null,
    error: null,
    isAuthenticated: false,
    userData: null,
  });
  const [confirmModal, setConfirmModal] = useState({
    open: false,
    title: '',
    message: '',
    onConfirm: () => { },
  });

  const showConfirmModal = (title, message, onConfirm) => {
    setConfirmModal({ open: true, title, message, onConfirm });
  };

  const closeConfirmModal = () => {
    setConfirmModal({
      open: false,
      title: '',
      message: '',
      onConfirm: () => { },
    });
  };

  const handleLogout = useCallback(() => {
    localStorage.removeItem('token');
    setAppState({
      isLoading: false,
      fileInfo: null,
      analysisData: null,
      error: null,
      isAuthenticated: false,
      userData: null,
    });
    navigate('/login');
  }, [navigate]);

  const validateToken = useCallback(
    async (token) => {
      try {
        const response = await fetch(
          'http://localhost:8080/api/validate-token',
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        if (response.ok) {
          setAppState((prev) => ({ ...prev, isAuthenticated: true }));
        } else {
          handleLogout();
        }
      } catch {
        handleLogout();
      }
    },
    [handleLogout]
  );

  const fetchUserData = useCallback(async (token) => {
    try {
      const response = await fetch('http://localhost:8080/api/user', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setAppState((prev) => ({ ...prev, userData: data }));
      }
    } catch (err) {
      console.error('Error fetching user data:', err);
    }
  }, []);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      validateToken(token);
      fetchUserData(token);
    }
  }, [validateToken, fetchUserData]);

  const handleFileUpload = async (file) => {
    if (!file) return;
    const abortController = new AbortController();
    abortControllerRef.current = abortController;
    setAppState((prev) => ({
      ...prev,
      isLoading: true,
      error: null,
      fileInfo: { name: file.name, size: file.size },
    }));

    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Требуется авторизация');
      const formData = new FormData();
      formData.append('document', file);

      const response = await fetch('http://localhost:8080/api/analyze', {
        method: 'POST',
        body: formData,
        headers: { Authorization: `Bearer ${token}` },
        signal: abortController.signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Ошибка при анализе документа');
      }

      const data = await response.json();
      setAppState((prev) => ({
        ...prev,
        isLoading: false,
        analysisData: data,
      }));
    } catch (err) {
      if (err.name === 'AbortError') {
        console.warn('Анализ отменен пользователем');
        setAppState((prev) => ({
          ...prev,
          isLoading: false,
          fileInfo: null,
          analysisData: null,
          error: 'Анализ отменен пользователем',
        }));
      } else {
        setAppState((prev) => ({
          ...prev,
          isLoading: false,
          error: err.message,
        }));
        if (
          err.message.includes('Сессия истекла') ||
          err.message.includes('Требуется авторизация')
        ) {
          handleLogout();
        }
      }
    }
  };

  const handleCancelUpload = () => {
    showConfirmModal(
      'Отменить анализ',
      'Вы уверены, что хотите отменить анализ документа?',
      () => {
        if (abortControllerRef.current) abortControllerRef.current.abort();
      }
    );
  };

  const handleRemoveFile = () => {
    showConfirmModal(
      'Удалить файл',
      'Вы уверены, что хотите удалить загруженный файл?',
      async () => {
        try {
          const token = localStorage.getItem('token');
          const response = await fetch(
            'http://localhost:8080/api/cache/clear',
            {
              method: 'POST',
              headers: { Authorization: `Bearer ${token}` },
            }
          );
          if (!response.ok) throw new Error('Не удалось удалить файл');

          setAppState((prev) => ({ ...prev, fileInfo: null, error: null }));
        } catch (err) {
          setAppState((prev) => ({ ...prev, error: err.message }));
        }
      }
    );
  };

  return (
    <div className="App">
      {appState.isAuthenticated && (
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            background: '#fff',
            borderBottom: '1px solid #E5E7EB',
            padding: '0.5rem 2rem',
            position: 'sticky',
            top: 0,
            zIndex: 1500,
          }}
        >
          <Box sx={{ display: 'flex', gap: 2 }}>
            <Button
              startIcon={<HomeIcon sx={{ fontSize: '1.2rem' }} />}
              onClick={() => navigate('/')}
              sx={{ textTransform: 'none', color: '#000000', '&:hover': { color: '#E60000', bgcolor: 'transparent' } }}
            >
              Главная
            </Button>
            <Button
              startIcon={<HistoryIcon sx={{ fontSize: '1.2rem' }} />}
              onClick={() => navigate('/history')}
              sx={{ textTransform: 'none', color: '#000000', '&:hover': { color: '#E60000', bgcolor: 'transparent' } }}
            >
              История
            </Button>
            <Button
              startIcon={<ProfileIcon sx={{ fontSize: '1.2rem' }} />}
              onClick={() => navigate('/profile')}
              sx={{ textTransform: 'none', color: '#000000', '&:hover': { color: '#E60000', bgcolor: 'transparent' } }}
            >
              Профиль
            </Button>
            <Button
              startIcon={<ChatIcon sx={{ fontSize: '1.2rem' }} />}
              onClick={() => navigate('/chat')}
              sx={{ textTransform: 'none', color: '#000000', '&:hover': { color: '#E60000', bgcolor: 'transparent' } }}
            >
              Консультация
            </Button>
            {appState.userData?.role === 'admin' && (
              <Button
                startIcon={<EvalIcon sx={{ fontSize: '1.2rem' }} />}
                onClick={() => navigate('/admin/eval')}
                sx={{ textTransform: 'none', color: '#000000', '&:hover': { color: '#E60000', bgcolor: 'transparent' } }}
              >
                HITL Admin
              </Button>
            )}
            {(appState.userData?.role === 'admin' || appState.userData?.role === 'professor' || appState.userData?.role === 'student') && (
              <Button
                startIcon={<EvalIcon sx={{ fontSize: '1.2rem' }} />}
                onClick={() => navigate('/reviewer/eval')}
                sx={{ textTransform: 'none', color: '#000000', '&:hover': { color: '#E60000', bgcolor: 'transparent' } }}
              >
                HITL Review
              </Button>
            )}
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Button
              startIcon={
                <Avatar
                  sx={{
                    width: 32,
                    height: 32,
                    bgcolor: '#333333',
                  }}
                >
                  {appState.userData?.email?.charAt(0).toUpperCase() || 'U'}
                </Avatar>
              }
              onClick={() => navigate('/profile')}
              sx={{
                color: 'text.primary',
                textTransform: 'none',
                '&:hover': {
                  backgroundColor: 'rgba(0, 0, 0, 0.04)',
                },
              }}
            >
              {appState.userData?.email || 'Профиль'}
            </Button>
            <Button
              variant="outlined"
              onClick={handleLogout}
              sx={{
                textTransform: 'none',
                color: '#E60000',
                borderColor: '#E60000',
                '&:hover': { borderColor: '#CC0000', backgroundColor: 'rgba(230,0,0,0.05)' }
              }}
            >
              Выход
            </Button>
          </Box>
        </Box>
      )}

      <Header
        isAuthenticated={appState.isAuthenticated}
        onLogout={handleLogout}
        userData={appState.userData}
      />

      <CContainer fluid className="main-content py-4 modern-container">
        <Routes>
          <Route
            path="/login"
            element={
              <AuthPage
                onSuccess={() => {
                  const token = localStorage.getItem('token');
                  setAppState((prev) => ({ ...prev, isAuthenticated: true }));
                  fetchUserData(token);
                  navigate('/');
                }}
                type="login"
              />
            }
          />
          <Route
            path="/register"
            element={
              <AuthPage onSuccess={() => navigate('/login')} type="register" />
            }
          />
          <Route
            path="/auth/callback"
            element={
              <AuthCallback
                onSuccess={() => {
                  const token = localStorage.getItem('token');
                  setAppState((prev) => ({ ...prev, isAuthenticated: true }));
                  fetchUserData(token);
                  navigate('/');
                }}
              />
            }
          />
          <Route
            path="/profile"
            element={
              <ProtectedRoute isAuthenticated={appState.isAuthenticated}>
                <ProfileSection
                  userData={appState.userData}
                  onLogout={handleLogout}
                  onBack={() => navigate('/')}
                />
              </ProtectedRoute>
            }
          />
          <Route
            path="/chat"
            element={
              <ProtectedRoute isAuthenticated={appState.isAuthenticated}>
                <Box sx={{ display: 'flex', height: 'calc(100vh - 64px)', overflow: 'hidden' }}>
                  <Sidebar
                    sessions={chatHistory.sessions}
                    activeSessionId={chatHistory.activeSessionId}
                    onSelectSession={chatHistory.setActiveSessionId}
                    onNewChat={chatHistory.createNewSession}
                    onDeleteSession={chatHistory.deleteSession}
                  />
                  <Box sx={{ flex: 1, height: '100%', overflow: 'hidden' }}>
                    <ChatSection
                      activeSession={chatHistory.activeSession}
                      addMessageToSession={chatHistory.addMessageToSession}
                      onClearHistory={(sessionId) => {
                        if (window.confirm('Очистить историю этого чата?')) {
                          chatHistory.addMessageToSession(sessionId, {
                            content: 'История очищена!',
                            isUser: false,
                            mode: 'legal_rag'
                          });
                        }
                      }}
                      onExportChat={(session) => {
                        const csvContent = "data:text/csv;charset=utf-8,Role,Message\n"
                          + session.messages.map(m => `${m.isUser ? 'User' : 'Assistant'},"${m.content.replace(/"/g, '""')}"`).join("\n");
                        const link = document.createElement("a");
                        link.setAttribute("href", encodeURI(csvContent));
                        link.setAttribute("download", `chat_${session.title}.csv`);
                        document.body.appendChild(link);
                        link.click();
                      }}
                    />
                  </Box>
                </Box>
              </ProtectedRoute>
            }
          />
          <Route
            path="/history"
            element={
              <ProtectedRoute isAuthenticated={appState.isAuthenticated}>
                <HistorySection onBackClick={() => navigate('/')} />
              </ProtectedRoute>
            }
          />
          <Route
            path="/admin/eval"
            element={
              <AdminRoute userData={appState.userData}>
                <EvaluationAdmin />
              </AdminRoute>
            }
          />
          <Route
            path="/reviewer/eval"
            element={
              <ReviewerRoute userData={appState.userData}>
                <ReviewerDashboard />
              </ReviewerRoute>
            }
          />
          <Route
            path="/reviewer/eval/:taskId"
            element={
              <ReviewerRoute userData={appState.userData}>
                <EvaluationForm />
              </ReviewerRoute>
            }
          />
          <Route
            path="/"
            element={
              <ProtectedRoute isAuthenticated={appState.isAuthenticated}>
                {appState.isLoading ? (
                  <LoadingSection onCancel={handleCancelUpload} />
                ) : appState.analysisData ? (
                  <ResultSection
                    data={appState.analysisData}
                    onBackClick={() =>
                      setAppState((prev) => ({
                        ...prev,
                        analysisData: null,
                      }))
                    }
                  />
                ) : (
                  <UploadSection
                    onFileUpload={handleFileUpload}
                    fileInfo={appState.fileInfo}
                    error={appState.error}
                    onClearError={() =>
                      setAppState((prev) => ({ ...prev, error: null }))
                    }
                    onHistoryClick={() => navigate('/history')}
                    onCancelUpload={handleCancelUpload}
                    onRemoveFile={handleRemoveFile}
                  />
                )}
              </ProtectedRoute>
            }
          />
        </Routes>
      </CContainer>

      <Footer />

      <Modal
        open={confirmModal.open}
        onClose={closeConfirmModal}
        closeAfterTransition
      >
        <Fade in={confirmModal.open}>
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              width: 400,
              bgcolor: 'background.paper',
              boxShadow: 24,
              p: 4,
              borderRadius: 3,
            }}
          >
            <Typography variant="h6">{confirmModal.title}</Typography>
            <Typography sx={{ mt: 2 }}>{confirmModal.message}</Typography>
            <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
              <Button onClick={closeConfirmModal} sx={{ mr: 2 }}>
                Отмена
              </Button>
              <Button
                variant="contained"
                onClick={() => {
                  confirmModal.onConfirm();
                  closeConfirmModal();
                }}
                sx={{
                  bgcolor: '#E60000',
                  '&:hover': { bgcolor: '#CC0000' },
                  textTransform: 'none'
                }}
              >
                Подтвердить
              </Button>
            </Box>
          </Box>
        </Fade>
      </Modal>
    </div>
  );
}

export default AppWrapper;
