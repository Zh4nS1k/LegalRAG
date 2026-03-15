package controllers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"legally/api/middleware"
	"legally/models"
	"legally/services"
	"legally/utils"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// MAX_HISTORY_MESSAGES is the number of recent messages sent to the AI as context.
// Each user–assistant exchange = 2 messages, so 20 = last 10 turns.
const MAX_HISTORY_MESSAGES = 20

// aiHTTPClient is re-used across all chat requests to the Python AI engine.
// A persistent client avoids re-establishing TCP connections on every request.
var aiHTTPClient = &http.Client{
	Timeout: 180 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        10,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		DisableKeepAlives:   false,
	},
}

type HistoryMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Message string `json:"message" binding:"required"`
	// History is ignored server-side; real history is loaded from MongoDB per user.
	// Kept for API backward compatibility only.
	History []HistoryMessage `json:"history"`
}

type PythonChatRequest struct {
	Query   string           `json:"query"`
	History []HistoryMessage `json:"history"`
}

// Structs to match the Python API response
type PythonSourceDocument struct {
	PageContent string                 `json:"page_content"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type PythonChatResponse struct {
	Result               string                 `json:"result"`
	SourceDocuments      []PythonSourceDocument `json:"source_documents"`
	TraceReport          map[string]interface{} `json:"trace_report"`
	ConfidenceScore      float64                `json:"confidence_score"`
	MissingFields        []string               `json:"missing_fields"`
	ClarifyingQuestions  []string               `json:"clarifying_questions"`
}

// Structs for the Frontend response (matching what ChatSection.js expects)
// Frontend expects: { answer: string, mode: string, sources: []SourceDetail }
// Detective Mode adds: confidence_score, missing_fields, clarifying_questions
type ChatResponse struct {
	Answer               string                `json:"answer"`
	Mode                 string                `json:"mode"`
	Sources              []models.SourceDetail `json:"sources"`
	TraceReport          interface{}           `json:"trace_report,omitempty"`
	ConfidenceScore      float64               `json:"confidence_score,omitempty"`
	MissingFields        []string              `json:"missing_fields,omitempty"`
	ClarifyingQuestions  []string              `json:"clarifying_questions,omitempty"`
}

func HandleChat(c *gin.Context) {
	// Authentication check (redundant if middleware is used, but good for safety)
	_, exists := c.Get("userId")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Authentication required"})
		return
	}

	startVal := time.Now()
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Message is required"})
		return
	}
	middleware.RecordMetric(c, "request_validation", time.Since(startVal))

	utils.LogInfo(fmt.Sprintf("Received chat request: %s", req.Message))

	// Get userId from context
	userID := c.MustGet("userId").(string)

	// Save User Message to MongoDB
	if err := services.SaveChatMessage(userID, "user", req.Message, nil); err != nil {
		utils.LogError(fmt.Sprintf("Failed to save user message: %v", err))
		// Continue processing even if save fails
	}

	// ── SERVER-SIDE HISTORY LOADING ──────────────────────────────────────────
	// Load the last MAX_HISTORY_MESSAGES for THIS user from MongoDB.
	// This guarantees per-user isolation regardless of what the client sends.
	dbMessages, histErr := services.GetRecentChatHistory(userID, MAX_HISTORY_MESSAGES)
	serverHistory := make([]HistoryMessage, 0, len(dbMessages))
	if histErr != nil {
		utils.LogWarning(fmt.Sprintf("Could not load chat history for %s: %v", userID, histErr))
		// Continue with empty history — better than blocking the request
	} else {
		for _, msg := range dbMessages {
			// Skip the message we just saved (the current user turn)
			// to avoid echoing it back as context
			if msg.Role == "user" && msg.Content == req.Message {
				continue
			}
			serverHistory = append(serverHistory, HistoryMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}
	// ─────────────────────────────────────────────────────────────────────────

	// Prepare request to Python API using server-loaded history
	pythonPayload := PythonChatRequest{
		Query:   req.Message,
		History: serverHistory, // ← always from MongoDB, never from client
	}
	jsonData, err := json.Marshal(pythonPayload)
	if err != nil {
		utils.LogError(fmt.Sprintf("Failed to marshal python payload: %v", err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}

	// Python API URL (use AI_SERVICE_URL in Docker, e.g. http://ai_service:8000)
	pythonAPIURL := utils.GetAIServiceBaseURL() + "/api/v1/internal-chat"

	startInternal := time.Now()
	httpReq, err := http.NewRequest("POST", pythonAPIURL, bytes.NewBuffer(jsonData))
	if err != nil {
		utils.LogError(fmt.Sprintf("Failed to create request: %v", err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal server error"})
		return
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if traceID, exists := c.Get("X-Trace-ID"); exists {
		httpReq.Header.Set("X-Trace-ID", traceID.(string))
	}

	// Use the persistent aiHTTPClient — avoids a new TCP handshake on every request
	resp, err := aiHTTPClient.Do(httpReq)
	middleware.RecordMetric(c, "internal_service_call", time.Since(startInternal))
	if err != nil {
		utils.LogError(fmt.Sprintf("Failed to call Python API: %v", err))
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "ИИ-сервис недоступен. Попробуйте позже."})
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		errorMsg := string(bodyBytes)
		utils.LogError(fmt.Sprintf("Python API returned error: %d - %s", resp.StatusCode, errorMsg))
		
		// Map specific internal errors to user-friendly messages
		if strings.Contains(errorMsg, "Rate limit reached") {
			c.JSON(http.StatusTooManyRequests, gin.H{"error": "Превышен лимит запросов к ИИ. Пожалуйста, попробуйте через несколько минут."})
		} else {
			c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Ошибка ИИ-сервиса при обработке вопроса."})
		}
		return
	}

	// Parse Python response
	var pythonResp PythonChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&pythonResp); err != nil {
		utils.LogError(fmt.Sprintf("Failed to decode Python response: %v", err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process AI response"})
		return
	}

	// Transform to Frontend format
	sources := make([]models.SourceDetail, 0)
	for _, doc := range pythonResp.SourceDocuments {
		// Format source string, e.g., "Source Name (Article 123)"
		sourceTitle := ""
		if src, ok := doc.Metadata["source"].(string); ok {
			sourceTitle += src
		}
		if art, ok := doc.Metadata["article_number"]; ok {
			sourceTitle += fmt.Sprintf(" (ст. %v)", art)
		}
		if sourceTitle == "" {
			sourceTitle = "Unknown Source"
		}
		sources = append(sources, models.SourceDetail{
			Title: sourceTitle,
			Text:  doc.PageContent,
		})
	}

	startDB := time.Now()
	// Save AI response
	_ = services.SaveChatMessage(userID, "assistant", pythonResp.Result, sources)
	middleware.RecordMetric(c, "db_cache_overhead", time.Since(startDB))

	var finalTraceReport interface{}
	if timerObj, exists := c.Get("latency_timer"); exists {
		if timer, ok := timerObj.(*middleware.Timer); ok {
			goProcessing := time.Since(timer.StartTime).Milliseconds()

			if pythonResp.TraceReport != nil {
				if metricsMs, ok := pythonResp.TraceReport["metrics_ms"].(map[string]interface{}); ok {
					metricsMs["go_processing"] = goProcessing
					if breakdown, ok := metricsMs["breakdown"].(map[string]interface{}); ok {
						for k, v := range timer.Metrics {
							breakdown[k] = v
						}
					}
				}
				finalTraceReport = pythonResp.TraceReport
			}
		}
	}

	response := ChatResponse{
		Answer:              pythonResp.Result,
		Mode:                "legal_rag",
		Sources:             sources,
		TraceReport:         finalTraceReport,
		ConfidenceScore:     pythonResp.ConfidenceScore,
		MissingFields:       pythonResp.MissingFields,
		ClarifyingQuestions: pythonResp.ClarifyingQuestions,
	}

	c.JSON(http.StatusOK, response)
}

func GetChatHistory(c *gin.Context) {
	userID, exists := c.Get("userId")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Authentication required"})
		return
	}

	history, err := services.GetChatHistory(userID.(string))
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка получения истории чата: %v", err))
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch chat history"})
		return
	}

	c.JSON(http.StatusOK, history)
}

func ClearChatHistory(c *gin.Context) {
	userID, exists := c.Get("userId")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Authentication required"})
		return
	}

	if err := services.ClearChatHistory(userID.(string)); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to clear chat history"})
		return
	}

	c.JSON(http.StatusOK, gin.H{"success": true})
}

func ExportChatHistory(c *gin.Context) {
	userID, exists := c.Get("userId")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Authentication required"})
		return
	}

	data, err := services.ExportChatHistory(userID.(string))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to export chat history"})
		return
	}

	c.Header("Content-Disposition", "attachment; filename=chat_history.csv")
	c.Header("Content-Type", "text/csv")
	c.Data(http.StatusOK, "text/csv", data)
}
