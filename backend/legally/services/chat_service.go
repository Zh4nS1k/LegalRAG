package services

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"legally/models"
	"legally/repositories"
	"net/http"
	"strings"
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

// aiHTTPClient is a persistent singleton — reuses TCP connections to the Python AI engine.
// Creating a new http.Client per request wastes ~100–200ms on TLS handshake to localhost.
var aiHTTPClient = &http.Client{
	Timeout: 300 * time.Second,
	Transport: &http.Transport{
		MaxIdleConns:        10,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		DisableKeepAlives:   false,
	},
}

func SaveChatMessage(userID, role, content string, sources []models.SourceDetail) error {
	objID, err := primitive.ObjectIDFromHex(userID)
	if err != nil {
		return fmt.Errorf("invalid user id")
	}

	msg := models.ChatMessage{
		UserID:    objID,
		Role:      role,
		Content:   content,
		Sources:   sources,
		CreatedAt: time.Now(),
	}

	return repositories.SaveChatMessage(msg)
}

func GetChatHistory(userID string) ([]models.ChatMessage, error) {
	return repositories.GetChatHistory(userID)
}

// GetRecentChatHistory returns the last `limit` messages for the given user.
// Used by HandleChat to build server-side history for the Python AI request.
func GetRecentChatHistory(userID string, limit int) ([]models.ChatMessage, error) {
	return repositories.GetRecentChatHistory(userID, limit)
}

func ClearChatHistory(userID string) error {
	return repositories.ClearChatHistory(userID)
}

func ExportChatHistory(userID string) ([]byte, error) {
	history, err := repositories.GetChatHistory(userID)
	if err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	// Write BOM for Excel compatibility with UTF-8
	buf.Write([]byte{0xEF, 0xBB, 0xBF})

	writer := csv.NewWriter(&buf)
	
	// Create header
	header := []string{"Дата", "Роль", "Сообщение", "Источники"}
	if err := writer.Write(header); err != nil {
		return nil, err
	}

	for _, msg := range history {
		sourcesStr := ""
		if len(msg.Sources) > 0 {
			titles := make([]string, len(msg.Sources))
			for i, s := range msg.Sources {
				titles[i] = s.Title
			}
			sourcesStr = strings.Join(titles, "; ")
		}
		
		roleName := "Пользователь"
		if msg.Role == "assistant" {
			roleName = "AI Ассистент"
		}

		record := []string{
			msg.CreatedAt.Format("2006-01-02 15:04:05"),
			roleName,
			msg.Content,
			sourcesStr,
		}
		
		if err := writer.Write(record); err != nil {
			return nil, err
		}
	}
	
	writer.Flush()
	return buf.Bytes(), nil
}
