package utils

import "os"

// GetAIServiceBaseURL returns the base URL of the AI (Python) service.
// In Docker Compose use AI_SERVICE_URL=http://ai_service:8000 so the backend container can reach the ai_service container.
// For local dev, default is http://localhost:8000 (or set AI_SERVICE_URL=http://localhost:8001 if ai_service runs on 8001).
func GetAIServiceBaseURL() string {
	if u := os.Getenv("AI_SERVICE_URL"); u != "" {
		return u
	}
	return "http://localhost:8000"
}
