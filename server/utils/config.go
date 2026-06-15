package utils

import (
	"net/url"
	"os"
)

// GetAIServiceBaseURL returns the base URL of the AI (Python) service.
// In Docker Compose use AI_SERVICE_URL=http://ai_service:8000 so the backend container can reach the ai_service container.
// For local dev, default is http://localhost:8000 (or set AI_SERVICE_URL=http://localhost:8001 if ai_service runs on 8001).
func GetAIServiceBaseURL() string {
	if u := os.Getenv("AI_SERVICE_URL"); u != "" {
		// If backend is running on host (not in Docker) and AI_SERVICE_URL points
		// to the Compose service name, it won't resolve. Fallback to localhost.
		if !runningInDocker() {
			if parsed, err := url.Parse(u); err == nil && parsed.Hostname() == "ai_service" {
				parsed.Host = "127.0.0.1:8000"
				return parsed.String()
			}
		}
		return u
	}
	return "http://localhost:8000"
}

func runningInDocker() bool {
	// Present by default inside Docker containers.
	if _, err := os.Stat("/.dockerenv"); err == nil {
		return true
	}
	return false
}
