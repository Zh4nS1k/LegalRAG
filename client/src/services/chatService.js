const BACKEND_BASE_URL = (process.env.REACT_APP_BACKEND_URL || 'http://localhost:8080').replace(/\/+$/, '');

export function normalizeChatResponse(payload) {
  const response = payload && typeof payload === 'object' ? payload : {};
  return {
    answer: response.answer || response.result || '',
    mode: response.mode || 'legal_rag',
    sources: Array.isArray(response.sources) ? response.sources : (Array.isArray(response.source_documents) ? response.source_documents : []),
    trace_report: response.trace_report || null,
    confidence_score: response.confidence_score ?? null,
    missing_fields: Array.isArray(response.missing_fields) ? response.missing_fields : [],
    clarifying_questions: Array.isArray(response.clarifying_questions) ? response.clarifying_questions : [],
  };
}

export async function sendChatMessage({ token, message, chatId, history }) {
  const response = await fetch(`${BACKEND_BASE_URL}/api/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      message,
      chat_id: chatId,
      history,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    const error = new Error(
      response.status === 401
        ? 'Сессия истекла. Пожалуйста, войдите снова.'
        : (errorData.error || 'Ошибка при обработке запроса')
    );
    error.status = response.status;
    throw error;
  }

  return normalizeChatResponse(await response.json());
}
