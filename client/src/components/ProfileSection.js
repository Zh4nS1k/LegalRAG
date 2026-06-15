// components/ProfileSection.js
import React from 'react';
import {
  Container,
  Typography,
  Button,
  Paper,
  Avatar,
  Box,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  Person,
  Email,
  CalendarToday,
  ExitToApp,
  ArrowBack,
} from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import { useNavigate } from 'react-router-dom';

const ProfilePaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(4),
  borderRadius: '16px',
  border: '1px solid #E5E7EB',
  maxWidth: 600,
  margin: '0 auto',
  position: 'relative',
  boxShadow: 'none',
}));

function ProfileSection({ userData, onLogout }) {
  const navigate = useNavigate();

  if (!userData) {
    navigate('/login');
    return null;
  }

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Box sx={{ mb: 2 }}>
        <Button
          startIcon={<ArrowBack />}
          onClick={() => navigate('/')}
          sx={{ color: 'text.secondary' }}
        >
          Назад
        </Button>
      </Box>

      <ProfilePaper elevation={0}>
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 600 }}>
          Профиль пользователя
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <Avatar
            sx={{ width: 80, height: 80, mr: 3, bgcolor: 'primary.main' }}
          >
            <Person sx={{ fontSize: 40 }} />
          </Avatar>
          <Typography variant="h5">{userData.email}</Typography>
        </Box>

        <Divider sx={{ my: 2 }} />

        <List>
          <ListItem>
            <ListItemIcon>
              <Email />
            </ListItemIcon>
            <ListItemText primary="Email" secondary={userData.email} />
          </ListItem>
          <ListItem>
            <ListItemIcon>
              <CalendarToday />
            </ListItemIcon>
            <ListItemText
              primary="Дата регистрации"
              secondary={new Date(userData.createdAt).toLocaleDateString()}
            />
          </ListItem>
        </List>

        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'space-between' }}>
          <Button
            variant="outlined"
            startIcon={<ArrowBack />}
            onClick={() => navigate('/')}
          >
            Назад к загрузке
          </Button>
          <Button
            variant="contained"
            color="error"
            startIcon={<ExitToApp />}
            onClick={onLogout}
            size="large"
          >
            Выйти
          </Button>
        </Box>
      </ProfilePaper>
    </Container>
  );
}

export default ProfileSection;
