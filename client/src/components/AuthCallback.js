import React, { useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Box, CircularProgress, Typography } from '@mui/material';

export default function AuthCallback({ onSuccess }) {
  const navigate = useNavigate();
  const location = useLocation();
  const handledRef = useRef(false);

  useEffect(() => {
    // Run only once per mount when URL has tokens; avoid re-running when parent re-renders (onSuccess changes)
    if (handledRef.current) return;
    const searchParams = new URLSearchParams(location.search);
    const accessToken = searchParams.get('accessToken');
    const refreshToken = searchParams.get('refreshToken');

    if (accessToken) {
      handledRef.current = true;
      localStorage.setItem('token', accessToken);
      if (refreshToken) {
        localStorage.setItem('refreshToken', refreshToken);
      }
      onSuccess();
    } else {
      console.error('No access token found in URL');
      navigate('/login');
    }
  }, [location.search, navigate, onSuccess]);

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh' }}>
      <CircularProgress sx={{ mb: 2, color: '#E60000' }} />
      <Typography>Авторизация...</Typography>
    </Box>
  );
}
