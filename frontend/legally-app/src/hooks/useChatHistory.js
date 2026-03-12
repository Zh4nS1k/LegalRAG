import { useState, useEffect, useCallback } from 'react';

// Extract userId from JWT token to securely namespace the chat history per user
const getStorageKey = () => {
    try {
        const token = localStorage.getItem('token');
        if (token) {
            // Decode the payload part of the JWT (it's base64 encoded JSON)
            const payload = JSON.parse(atob(token.split('.')[1]));
            return `legally_chat_sessions_${payload.user_id}`;
        }
    } catch (e) {
        console.error("Failed to parse JWT for chat storage key", e);
    }
    return 'legally_chat_sessions'; // fallback for unauthenticated
};

export const useChatHistory = () => {
    const [sessions, setSessions] = useState(() => {
        const saved = localStorage.getItem(getStorageKey());
        return saved ? JSON.parse(saved) : [];
    });
    const [activeSessionId, setActiveSessionId] = useState(null);

    // If the token changes (e.g., user logs in/out), we should update the storage key.
    // For simplicity, we save based on the *current* token's user.
    useEffect(() => {
        localStorage.setItem(getStorageKey(), JSON.stringify(sessions));
    }, [sessions]);

    const createNewSession = useCallback(() => {
        const newSession = {
            id: Date.now().toString(),
            title: 'Новый чат',
            messages: [],
            createdAt: new Date().toISOString(),
        };
        setSessions(prev => [newSession, ...prev]);
        setActiveSessionId(newSession.id);
        return newSession.id;
    }, []);

    const addMessageToSession = useCallback((sessionId, message) => {
        setSessions(prev => prev.map(session => {
            if (session.id === sessionId) {
                let newTitle = session.title;
                // Update title from first user message
                if (session.messages.length === 0 && message.isUser) {
                    newTitle = message.content.slice(0, 30) + (message.content.length > 30 ? '...' : '');
                }
                return {
                    ...session,
                    title: newTitle,
                    messages: [...session.messages, message],
                };
            }
            return session;
        }));
    }, []);

    const deleteSession = useCallback((sessionId) => {
        setSessions(prev => prev.filter(s => s.id !== sessionId));
        if (activeSessionId === sessionId) {
            setActiveSessionId(null);
        }
    }, [activeSessionId]);

    const activeSession = sessions.find(s => s.id === activeSessionId);

    return {
        sessions,
        activeSessionId,
        setActiveSessionId,
        activeSession,
        createNewSession,
        addMessageToSession,
        deleteSession,
    };
};
