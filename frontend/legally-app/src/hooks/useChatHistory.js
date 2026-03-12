import { useState, useEffect, useCallback, useRef } from 'react';

// Extract userId from JWT token to securely namespace the chat history per user
const getStorageKey = () => {
    try {
        const token = localStorage.getItem('token');
        if (token) {
            // Decode the payload part of the JWT (it's base64 encoded JSON)
            const payload = JSON.parse(atob(token.split('.')[1]));
            // The Go backend struct uses `json:"userId"`
            if (payload.userId) {
                return `legally_chat_sessions_${payload.userId}`;
            }
        }
    } catch (e) {
        console.error("Failed to parse JWT for chat storage key", e);
    }
    return 'legally_chat_sessions'; // fallback for unauthenticated
};

export const useChatHistory = () => {
    // 1. Synchronously initialize from localStorage so we don't start with []
    const [storageKey, setStorageKey] = useState(getStorageKey());
    
    const [sessions, setSessions] = useState(() => {
        const saved = localStorage.getItem(getStorageKey());
        return saved ? JSON.parse(saved) : [];
    });
    
    const [activeSessionId, setActiveSessionId] = useState(() => {
        const saved = localStorage.getItem(getStorageKey());
        const loadedSessions = saved ? JSON.parse(saved) : [];
        return loadedSessions.length > 0 ? loadedSessions[0].id : null;
    });

    const isFirstMount = useRef(true);

    // 2. Listen for storage/token changes (e.g. cross-tab logins, or local token updates)
    useEffect(() => {
        const handleAuthChange = () => {
            const currentKey = getStorageKey();
            
            // ONLY perform state updates if the authentication identity actually changed
            if (currentKey !== storageKey) {
                setStorageKey(currentKey);
                
                const saved = localStorage.getItem(currentKey);
                const loadedSessions = saved ? JSON.parse(saved) : [];
                
                setSessions(loadedSessions);
                
                if (loadedSessions.length > 0) {
                    setActiveSessionId((prev) => {
                        const exists = loadedSessions.find(s => s.id === prev);
                        return exists ? prev : loadedSessions[0].id;
                    });
                } else {
                    setActiveSessionId(null);
                }
            }
        };

        handleAuthChange(); // Fire on mount

        window.addEventListener('storage', handleAuthChange);
        const interval = setInterval(handleAuthChange, 1000); // Safely poll for same-tab token changes
        
        return () => {
            window.removeEventListener('storage', handleAuthChange);
            clearInterval(interval);
        };
    }, [storageKey]);

    // 3. Persist sessions back to localStorage ONLY if data actually changed
    useEffect(() => {
        if (isFirstMount.current) {
            isFirstMount.current = false;
            return; // Skip writing to localStorage on the very first React render
        }
        localStorage.setItem(storageKey, JSON.stringify(sessions));
    }, [sessions, storageKey]);

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
