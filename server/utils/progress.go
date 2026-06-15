// progress.go

package utils

import (
	"fmt"
	"io"
	"os"
	"strings"
	"time"
)

type ProgressBar struct {
	total    int64
	current  int64
	progress float64
	start    time.Time
	writer   io.Writer
}

func NewProgressBar(total int64, description string) *ProgressBar {
	pb := &ProgressBar{
		total:  total,
		start:  time.Now(),
		writer: os.Stdout,
	}
	fmt.Printf("\r%s [%s] 0%%", description, strings.Repeat(" ", 50))
	return pb
}

func (p *ProgressBar) Write(buf []byte) (int, error) {
	n := len(buf)
	p.current += int64(n)
	p.progress = float64(p.current) / float64(p.total) * 100

	bar := strings.Repeat("=", int(p.progress/2))
	fmt.Printf("\r%s [%-50s] %3.0f%%",
		"Загрузка файла",
		bar,
		p.progress)

	return n, nil
}

func (p *ProgressBar) Finish() {
	fmt.Printf("\n✅ Завершено за %v\n", time.Since(p.start))
}
