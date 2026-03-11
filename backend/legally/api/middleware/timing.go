package middleware

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"time"
)

// Timer holds the latency metrics for the current request
type Timer struct {
	StartTime time.Time
	Metrics   map[string]int64
}

// TimingMiddleware extracts or generates X-Trace-ID and initializes metrics
func TimingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		traceID := c.GetHeader("X-Trace-ID")
		if traceID == "" {
			traceID = fmt.Sprintf("trace_%d", time.Now().UnixNano())
		}
		
		c.Set("X-Trace-ID", traceID)
		
		timer := &Timer{
			StartTime: time.Now(),
			Metrics:   make(map[string]int64),
		}
		c.Set("latency_timer", timer)
		
		c.Next()
	}
}

// RecordMetric safely records a processing duration step inside the ctx timer
func RecordMetric(c *gin.Context, step string, duration time.Duration) {
	if timerObj, exists := c.Get("latency_timer"); exists {
		if timer, ok := timerObj.(*Timer); ok {
			timer.Metrics[step] = duration.Milliseconds()
		}
	}
}
