// file_utils

package utils

import (
	"bytes"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/ledongthuc/pdf"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	maxFileSize    = 10 << 20 // 10MB
	tempFilePrefix = "temp_"
	pdfTimeout     = 30 * time.Second
)

func ProcessUploadedFile(c *gin.Context) (string, string, error) {
	LogAction("Начало обработки загруженного файла")

	// Ensure we don't process files that are too large
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxFileSize)
	if err := c.Request.ParseMultipartForm(maxFileSize); err != nil {
		LogError(fmt.Sprintf("Превышен максимальный размер файла (10MB): %v", err))
		return "", "", fmt.Errorf("размер файла не должен превышать 10MB")
	}

	file, header, err := c.Request.FormFile("document")
	if err != nil {
		LogError(fmt.Sprintf("Ошибка получения файла: %v", err))
		return "", "", fmt.Errorf("файл не получен")
	}
	defer file.Close()

	// Validate file extension
	ext := strings.ToLower(filepath.Ext(header.Filename))
	if ext != ".pdf" {
		LogError(fmt.Sprintf("Неподдерживаемый формат файла: %s", ext))
		return "", "", fmt.Errorf("поддерживаются только PDF файлы")
	}

	// Create temp file
	tempPath := filepath.Join("./temp", tempFilePrefix+fmt.Sprintf("%d_%s", time.Now().Unix(), header.Filename))
	LogInfo(fmt.Sprintf("Создание временного файла: %s", tempPath))

	tempFile, err := os.Create(tempPath)
	if err != nil {
		LogError(fmt.Sprintf("Ошибка создания временного файла: %v", err))
		return "", "", fmt.Errorf("ошибка создания временного файла")
	}

	// Copy file contents
	if _, err := io.Copy(tempFile, file); err != nil {
		tempFile.Close()
		LogError(fmt.Sprintf("Ошибка сохранения файла: %v", err))
		return "", "", fmt.Errorf("ошибка сохранения файла")
	}
	tempFile.Close()

	// Extract text from PDF
	text, err := SafeExtractTextFromPDF(tempPath, pdfTimeout)
	if err != nil {
		os.Remove(tempPath)
		LogError(fmt.Sprintf("Ошибка извлечения текста: %v", err))
		return "", "", fmt.Errorf("ошибка извлечения текста: %v", err)
	}

	if len(text) == 0 {
		os.Remove(tempPath)
		LogWarning("Документ не содержит текста")
		return "", "", fmt.Errorf("документ не содержит текста")
	}

	// Clean up temp file
	if err := os.Remove(tempPath); err != nil {
		LogWarning(fmt.Sprintf("Не удалось удалить временный файл: %v", err))
	}

	LogSuccess(fmt.Sprintf("Успешно обработан файл: %s (символов: %d)", header.Filename, len(text)))
	return text, header.Filename, nil
}

func SafeExtractTextFromPDF(path string, timeout time.Duration) (string, error) {
	result := make(chan string)
	errChan := make(chan error)

	go func() {
		text, err := ExtractTextFromPDF(path)
		if err != nil {
			errChan <- err
			return
		}
		result <- text
	}()

	select {
	case text := <-result:
		return text, nil
	case err := <-errChan:
		return "", err
	case <-time.After(timeout):
		return "", fmt.Errorf("таймаут извлечения текста (%v)", timeout)
	}
}

func ExtractTextFromPDF(path string) (string, error) {
	LogAction(fmt.Sprintf("Извлечение текста из PDF: %s", path))

	f, r, err := pdf.Open(path)
	if err != nil {
		return "", fmt.Errorf("не удалось открыть документ: %v", err)
	}
	defer f.Close()

	var buf bytes.Buffer
	b, err := r.GetPlainText()
	if err != nil {
		return "", fmt.Errorf("ошибка чтения документа: %v", err)
	}

	if _, err := io.Copy(&buf, b); err != nil {
		return "", fmt.Errorf("ошибка копирования текста: %v", err)
	}

	text := buf.String()
	if len(text) == 0 {
		return "", fmt.Errorf("файл пуст")
	}

	text = strings.Join(strings.Fields(text), " ")
	LogInfo(fmt.Sprintf("Извлечено %d символов из PDF", len(text)))
	return text, nil
}

func SplitText(text string, maxChars int) []string {
	LogAction(fmt.Sprintf("Разделение текста (макс. %d символов на часть)", maxChars))

	runes := []rune(text)
	var parts []string

	for start := 0; start < len(runes); start += maxChars {
		end := start + maxChars
		if end > len(runes) {
			end = len(runes)
		}
		parts = append(parts, string(runes[start:end]))
	}

	LogInfo(fmt.Sprintf("Текст разделен на %d частей", len(parts)))
	return parts
}
