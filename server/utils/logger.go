// logger.go

package utils

import (
	"fmt"
	"log"
	"time"
)

func LogInfo(message string) {
	log.Printf("ℹ️ %s", message)
}

func LogAction(message string) {
	log.Printf("🔄 %s", message)
}

func LogSuccess(message string) {
	log.Printf("✅ %s", message)
}

func LogWarning(message string) {
	log.Printf("⚠️ %s", message)
}

func LogError(message string) {
	log.Printf("❌ ERROR: %s", message)
}

func LogRequest(direction, url string, size int) {
	action := "➡️"
	if direction == "in" {
		action = "⬅️"
	}
	log.Printf("%s %s, длина тела: %d байт", action, url, size)
}

func WithTiming(action string, fn func()) {
	start := time.Now()
	LogAction(action)
	fn()
	LogSuccess(fmt.Sprintf("%s завершён за %v", action, time.Since(start)))
}
