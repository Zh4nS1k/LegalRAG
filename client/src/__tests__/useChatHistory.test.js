import { renderHook, act } from '@testing-library/react';
import { useChatHistory } from '../hooks/useChatHistory';

// Mock crypto.randomUUID
if (!global.crypto) {
  global.crypto = {
    randomUUID: () => 'test-uuid-' + Math.random(),
  };
}

describe('useChatHistory Hook', () => {
  beforeEach(() => {
    localStorage.clear();
    jest.clearAllMocks();
  });

  test('should create a new session with unique ID', () => {
    const { result } = renderHook(() => useChatHistory());

    act(() => {
      result.current.createNewSession();
    });

    expect(result.current.sessions.length).toBe(1);
    expect(result.current.activeSessionId).toMatch(/^test-uuid-/);
  });

  test('should change activeSessionId when creating new session', () => {
    const { result } = renderHook(() => useChatHistory());

    let firstId;
    act(() => {
      firstId = result.current.createNewSession();
    });

    let secondId;
    act(() => {
      secondId = result.current.createNewSession();
    });

    expect(firstId).not.toBe(secondId);
    expect(result.current.activeSessionId).toBe(secondId);
  });
});
