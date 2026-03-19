// AuthPage.js — Login / Register with field-level errors, OTP verification, role selector, Google OAuth

import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Container, Paper, Typography, TextField, Button, Box, Link, Divider,
  Fade, Alert, IconButton, InputAdornment, CircularProgress, MenuItem,
  Select, InputLabel, FormControl, FormHelperText,
} from '@mui/material';
import {
  Lock, Person, Email, Visibility, VisibilityOff, Google as GoogleIcon,
  CheckCircle,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import logo from '../images/hard_logo_legally.jpg';

const API = 'http://localhost:8080';

const AuthPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  borderRadius: theme.shape.borderRadius * 2,
  boxShadow: theme.shadows[6],
  maxWidth: 480,
  margin: '0 auto',
  position: 'relative',
  overflow: 'hidden',
  '&:before': {
    content: '""',
    position: 'absolute',
    top: 0, left: 0, right: 0,
    height: 4,
    background: 'linear-gradient(90deg, #E60000, #ff4444)',
  },
}));

const GoogleButton = styled(Button)(({ theme }) => ({
  borderColor: '#dadce0',
  color: '#3c4043',
  backgroundColor: '#fff',
  textTransform: 'none',
  fontWeight: 500,
  '&:hover': {
    backgroundColor: '#f8f9fa',
    borderColor: '#c6c6c6',
    boxShadow: '0 1px 3px rgba(60,64,67,0.2)',
  },
}));

const ROLE_OPTIONS = [
  { value: 'user',      label: 'Обычный пользователь' },
  { value: 'student',   label: 'Студент' },
  { value: 'professor', label: 'Преподаватель' },
];

// ─────────────────────────────────────────────────────────────────────────────

function AuthPage({ type, onSuccess }) {
  const navigate = useNavigate();
  const isLogin = type === 'login';

  // Form state
  const [form, setForm] = useState({
    email: '', password: '', name: '', role: 'user', showPassword: false,
  });

  // OTP state (after register)
  const [step, setStep] = useState('form'); // 'form' | 'verify'
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [otpError, setOtpError] = useState('');
  const [resendCooldown, setResendCooldown] = useState(0);
  const otpRefs = useRef([]);

  // Error state
  const [globalError, setGlobalError] = useState('');
  const [fieldErrors, setFieldErrors] = useState({}); // { email, password, role }
  const [isLoading, setIsLoading] = useState(false);
  const [successMsg, setSuccessMsg] = useState('');

  // ── Helpers ───────────────────────────────────────────────────────────────

  const clearErrors = () => { setGlobalError(''); setFieldErrors({}); };

  const switchMode = (to) => {
    clearErrors();
    setForm({ email: '', password: '', name: '', role: 'user', showPassword: false });
    setStep('form');
    navigate(to === 'login' ? '/login' : '/register');
  };

  const set = (key) => (e) => setForm(f => ({ ...f, [key]: e.target.value }));

  // ── Login / Register submit ───────────────────────────────────────────────

  const handleSubmit = async (e) => {
    e.preventDefault();
    clearErrors();

    // Client-side validation
    const errs = {};
    if (!form.email) errs.email = 'Введите email';
    if (!form.password) errs.password = 'Введите пароль'; // nosec
    if (!isLogin && !form.role) errs.role = 'Выберите роль';
    if (Object.keys(errs).length) { setFieldErrors(errs); return; }

    setIsLoading(true);
    try {
      const endpoint = isLogin ? '/api/login' : '/api/register';
      const body = isLogin
        ? { email: form.email, password: form.password }
        : { email: form.email, password: form.password, name: form.name, role: form.role };

      const res = await fetch(`${API}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      const data = await res.json();

      if (!res.ok) {
        // Map error_code to field-level errors
        const code = data.error_code;
        if (code === 'EMAIL_NOT_FOUND') {
          setFieldErrors({ email: 'Этот email не зарегистрирован' });
        } else if (code === 'WRONG_PASSWORD') {
          setFieldErrors({ password: 'Неверный пароль' });
        } else if (code === 'EMAIL_EXISTS') {
          setFieldErrors({ email: 'Этот email уже зарегистрирован' });
        } else {
          setGlobalError(data.error || 'Ошибка аутентификации');
        }
        return;
      }

      if (!isLogin && data.email_verified === false) {
        // Redirect to OTP verification step, DO NOT store tokens yet
        setStep('verify');
        startResendCooldown();
        return;
      }

      // If login or no verification needed, save token
      if (data.accessToken) {
        localStorage.setItem('token', data.accessToken);
      }

      onSuccess?.();
    } catch {
      setGlobalError('Не удалось подключиться к серверу. Попробуйте позже.');
    } finally {
      setIsLoading(false);
    }
  };

  // ── OTP ───────────────────────────────────────────────────────────────────

  const handleOtpChange = (idx, val) => {
    if (!/^\d?$/.test(val)) return; // digits only
    const next = [...otp];
    next[idx] = val;
    setOtp(next);
    setOtpError('');
    if (val && idx < 5) otpRefs.current[idx + 1]?.focus();
  };

  const handleOtpKeyDown = (idx, e) => {
    if (e.key === 'Backspace' && !otp[idx] && idx > 0) {
      otpRefs.current[idx - 1]?.focus();
    }
  };

  const handleOtpPaste = (e) => {
    const pasted = e.clipboardData.getData('text').replace(/\D/g, '').slice(0, 6);
    if (pasted.length === 6) {
      setOtp(pasted.split(''));
      otpRefs.current[5]?.focus();
    }
  };

  const handleVerify = async () => {
    const code = otp.join('');
    if (code.length < 6) { setOtpError('Введите все 6 цифр'); return; }
    setIsLoading(true);
    setOtpError('');
    try {
      const res = await fetch(`${API}/api/verify-code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: form.email, code }),
      });
      const data = await res.json();
      if (!res.ok) { setOtpError(data.error || 'Неверный код'); return; }
      setSuccessMsg('Email подтверждён! Входим...');
      setTimeout(() => onSuccess?.(), 1000);
    } catch {
      setOtpError('Ошибка соединения');
    } finally {
      setIsLoading(false);
    }
  };

  const startResendCooldown = () => {
    setResendCooldown(60);
    const t = setInterval(() => {
      setResendCooldown(c => { if (c <= 1) { clearInterval(t); return 0; } return c - 1; });
    }, 1000);
  };

  const handleResend = async () => {
    if (resendCooldown > 0) return;
    await fetch(`${API}/api/send-verification`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email: form.email }),
    });
    startResendCooldown();
  };

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────

  return (
    <Fade in timeout={600}>
      <Container maxWidth="sm" sx={{ py: 8 }}>
        <AuthPaper elevation={3}>

          {/* Logo + Title */}
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 3 }}>
            <Box
              component="img" src={logo} alt="Legally"
              sx={{
                width: 110, height: 110, mb: 2,
                filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.12))',
                animation: 'fadeInScale 0.8s ease-out',
                '@keyframes fadeInScale': {
                  '0%': { opacity: 0, transform: 'scale(0.8)' },
                  '100%': { opacity: 1, transform: 'scale(1)' },
                },
              }}
            />
            <Typography variant="h4" fontWeight={700} color="#000">
              {step === 'verify' ? 'Подтверждение email' : isLogin ? 'Вход в систему' : 'Регистрация'}
            </Typography>
            <Box sx={{ width: 40, height: 4, background: '#E60000', mt: 1 }} />
          </Box>

          {/* ── OTP verification step ── */}
          {step === 'verify' ? (
            <Box>
              <Typography color="text.secondary" align="center" mb={3}>
                Мы отправили 6-значный код на <strong>{form.email}</strong>
              </Typography>

              {successMsg && (
                <Alert icon={<CheckCircle />} severity="success" sx={{ mb: 2 }}>{successMsg}</Alert>
              )}
              {otpError && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setOtpError('')}>{otpError}</Alert>
              )}

              {/* 6 OTP boxes */}
              <Box sx={{ display: 'flex', gap: 1.5, justifyContent: 'center', mb: 3 }}>
                {otp.map((digit, i) => (
                  <TextField
                    key={i}
                    inputRef={el => (otpRefs.current[i] = el)}
                    value={digit}
                    onChange={e => handleOtpChange(i, e.target.value)}
                    onKeyDown={e => handleOtpKeyDown(i, e)}
                    onPaste={handleOtpPaste}
                    inputProps={{
                      maxLength: 1, style: { textAlign: 'center', fontSize: 24, fontWeight: 700 },
                    }}
                    sx={{ width: 52 }}
                    error={!!otpError}
                  />
                ))}
              </Box>

              <Button
                fullWidth variant="contained" size="large"
                onClick={handleVerify} disabled={isLoading}
                startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <CheckCircle />}
                sx={{ bgcolor: '#E60000', '&:hover': { bgcolor: '#CC0000' }, mb: 2 }}
              >
                Подтвердить
              </Button>

              <Typography align="center" color="text.secondary" fontSize={14}>
                Не получили?{' '}
                {resendCooldown > 0
                  ? `Повторная отправка через ${resendCooldown}с`
                  : <Link component="button" onClick={handleResend}>Отправить снова</Link>
                }
              </Typography>
            </Box>
          ) : (
            // ── Login / Register form ──
            <Box>
              {globalError && (
                <Alert severity="error" sx={{ mb: 2 }} onClose={() => setGlobalError('')}>{globalError}</Alert>
              )}

              {/* Google OAuth */}
              <GoogleButton
                fullWidth variant="outlined" size="large"
                startIcon={<GoogleIcon sx={{ color: '#4285F4' }} />}
                onClick={() => window.location.href = `${API}/api/auth/google`}
                sx={{ mb: 2 }}
              >
                {isLogin ? 'Войти через Google' : 'Зарегистрироваться через Google'}
              </GoogleButton>

              <Divider sx={{ mb: 2 }}>
                <Typography variant="caption" color="text.secondary">или</Typography>
              </Divider>

              <Box component="form" onSubmit={handleSubmit}>
                {/* Name — only on register */}
                {!isLogin && (
                  <TextField
                    fullWidth label="Имя (необязательно)" margin="normal"
                    value={form.name} onChange={set('name')}
                    InputProps={{ startAdornment: <InputAdornment position="start"><Person /></InputAdornment> }}
                  />
                )}

                {/* Email */}
                <TextField
                  fullWidth label="Email" type="email" margin="normal"
                  value={form.email} onChange={set('email')}
                  error={!!fieldErrors.email}
                  helperText={fieldErrors.email}
                  InputProps={{ startAdornment: <InputAdornment position="start"><Email /></InputAdornment> }}
                />

                {/* Password */}
                <TextField
                  fullWidth label="Пароль"
                  type={form.showPassword ? 'text' : 'password'}
                  margin="normal"
                  value={form.password} onChange={set('password')}
                  error={!!fieldErrors.password}
                  helperText={fieldErrors.password || (!isLogin ? 'Минимум 8 символов' : '')}
                  InputProps={{
                    startAdornment: <InputAdornment position="start"><Lock /></InputAdornment>,
                    endAdornment: (
                      <InputAdornment position="end">
                        <IconButton onClick={() => setForm(f => ({ ...f, showPassword: !f.showPassword }))}>
                          {form.showPassword ? <VisibilityOff /> : <Visibility />}
                        </IconButton>
                      </InputAdornment>
                    ),
                  }}
                />

                {/* Role selector — only on register */}
                {!isLogin && (
                  <FormControl fullWidth margin="normal" error={!!fieldErrors.role}>
                    <InputLabel>Роль *</InputLabel>
                    <Select value={form.role} label="Роль *" onChange={set('role')}>
                      {ROLE_OPTIONS.map(opt => (
                        <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                      ))}
                    </Select>
                    {fieldErrors.role && <FormHelperText>{fieldErrors.role}</FormHelperText>}
                  </FormControl>
                )}

                <Button
                  fullWidth variant="contained" size="large" type="submit"
                  disabled={isLoading}
                  startIcon={isLoading ? <CircularProgress size={20} color="inherit" /> : <Person />}
                  sx={{
                    mt: 3, bgcolor: '#E60000',
                    '&:hover': { bgcolor: '#CC0000', transform: 'translateY(-2px)' },
                    transition: 'all 0.3s ease',
                  }}
                >
                  {isLogin ? 'Войти' : 'Зарегистрироваться'}
                </Button>

                <Divider sx={{ my: 3 }} />

                <Typography align="center">
                  {isLogin ? (
                    <>Нет аккаунта?{' '}<Link component="button" onClick={() => switchMode('register')}>Зарегистрируйтесь</Link></>
                  ) : (
                    <>Уже есть аккаунт?{' '}<Link component="button" onClick={() => switchMode('login')}>Войдите</Link></>
                  )}
                </Typography>
              </Box>
            </Box>
          )}

        </AuthPaper>
      </Container>
    </Fade>
  );
}

export default AuthPage;
