import React from 'react';
import {
    Box,
    List,
    ListItem,
    ListItemButton,
    ListItemIcon,
    ListItemText,
    Typography,
    Button,
    IconButton,
    Divider,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import { styled } from '@mui/material/styles';

const SidebarContainer = styled(Box, {
    shouldForwardProp: (prop) => prop !== 'isOpen',
})(({ theme, isOpen }) => ({
    width: isOpen ? '280px' : '0px',
    height: '100%',
    backgroundColor: '#000000',
    color: '#FFFFFF',
    display: 'flex',
    flexDirection: 'column',
    borderRight: isOpen ? '2px solid #E60000' : 'none',
    overflow: 'hidden',
    transition: 'width 0.3s ease',
}));

const NewChatButton = styled(Button)(({ theme }) => ({
    width: 'calc(100% - 40px)',
    margin: '20px',
    padding: '12px',
    borderRadius: '12px',
    border: '1px solid #E60000',
    backgroundColor: '#E60000',
    color: 'white',
    textTransform: 'none',
    fontWeight: 600,
    justifyContent: 'center',
    transition: 'all 0.3s ease',
    '&:hover': {
        backgroundColor: '#CC0000',
        borderColor: '#CC0000',
        transform: 'translateY(-2px)'
    },
}));

const SessionListItem = styled(ListItem, {
    shouldForwardProp: (prop) => prop !== 'active',
})(({ theme, active }) => ({
    padding: '12px 12px',
    '&:hover .delete-btn': {
        opacity: 1,
    },
}));

const SessionButton = styled(ListItemButton, {
    shouldForwardProp: (prop) => prop !== 'active',
})(({ theme, active }) => ({
    borderRadius: '8px',
    margin: '4px 8px',
    backgroundColor: active ? 'transparent' : 'transparent',
    color: active ? '#FFFFFF' : '#9CA3AF',
    borderLeft: active ? '4px solid #E60000' : '4px solid transparent',
    transition: 'all 0.2s ease',
    '&:hover': {
        backgroundColor: 'transparent',
        color: '#E60000',
    },
}));

const Sidebar = ({ sessions, activeSessionId, onSelectSession, onNewChat, onDeleteSession, isOpen }) => {
    return (
        <SidebarContainer isOpen={isOpen}>
            <NewChatButton
                startIcon={<AddIcon />}
                onClick={onNewChat}
            >
                Новый чат
            </NewChatButton>

            <Typography
                variant="caption"
                sx={{ px: 3, py: 1, color: '#64748b', fontWeight: 'bold', textTransform: 'uppercase' }}
            >
                История чатов
            </Typography>

            <List sx={{ flex: 1, overflowY: 'auto', px: 1 }}>
                {sessions.map((session) => (
                    <SessionListItem
                        key={session.id}
                        disablePadding
                        secondaryAction={
                            <IconButton
                                edge="end"
                                className="delete-btn"
                                sx={{ opacity: 0, color: '#ef4444', p: 0.5, transition: 'opacity 0.2s' }}
                                onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteSession(session.id);
                                }}
                            >
                                <DeleteOutlineIcon fontSize="small" />
                            </IconButton>
                        }
                    >
                        <SessionButton
                            active={activeSessionId === session.id}
                            onClick={() => onSelectSession(session.id)}
                        >
                            <ListItemIcon sx={{ minWidth: 40, color: 'inherit' }}>
                                <ChatBubbleOutlineIcon fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                                primary={session.title}
                                primaryTypographyProps={{
                                    variant: 'body2',
                                    noWrap: true,
                                    sx: { fontSize: '0.875rem', fontWeight: (activeSessionId === session.id) ? 700 : 400 }
                                }}
                            />
                        </SessionButton>
                    </SessionListItem>
                ))}
            </List>

            <Divider sx={{ borderColor: '#333333' }} />
            <Box sx={{ p: 2, textAlign: 'center' }}>
                <Typography variant="caption" sx={{ color: '#666666', fontWeight: 600, letterSpacing: '0.1em' }}>
                    LEGALLY AI v1.0
                </Typography>
            </Box>
        </SidebarContainer>
    );
};

export default Sidebar;
