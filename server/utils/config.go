package utils

import (
	"net/url"
	"os"
)

// GetAIServiceBaseURL returns the base URL of the AI (Python) engine.
// In Docker Compose use AI_SERVICE_URL=http://engine:8000 so the backend container can reach the engine container.
// For local dev, default is http://localhost:8000 (or set AI_SERVICE_URL=http://localhost:8001 if engine runs on 8001).
func GetAIServiceBaseURL() string {
	if u := os.Getenv("AI_SERVICE_URL"); u != "" {
		// If backend is running on host (not in Docker) and AI_SERVICE_URL points
		// to the Compose service name, it won't resolve. Fallback to localhost.
		if !runningInDocker() {
			if parsed, err := url.Parse(u); err == nil && parsed.Hostname() == "engine" {
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
