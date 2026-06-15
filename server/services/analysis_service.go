package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"github.com/gin-gonic/gin"
	"golang.org/x/net/context"
	"io"
	"legally/repositories"
	"legally/utils"
	"net/http"
	"strings"
	"sync"
	"time"
)

func getAnalyzeEndpoint() string {
	return utils.GetAIServiceBaseURL() + "/api/v1/analyze"
}

type HttpError struct {
	Status  int
	Message string
}

func AnalyzeDocument(c *gin.Context) (interface{}, *HttpError) {
	utils.LogAction("Получен запрос на анализ документа")

	text, filename, err := utils.ProcessUploadedFile(c)
	if err != nil {
		utils.LogError(err.Error())
		return nil, &HttpError{Status: http.StatusBadRequest, Message: err.Error()}
	}

	utils.LogInfo(fmt.Sprintf("Извлечено %d символов из документа", len(text)))

	analysis, docType, err := AnalyzeText(text)
	if err != nil {
		utils.LogError(err.Error())
		return nil, &HttpError{Status: http.StatusInternalServerError, Message: err.Error()}
	}

	userID, _ := c.Get("userId")
	err = repositories.SaveAnalysis(userID.(string), filename, docType, analysis, text)
	if err != nil {
		utils.LogWarning(fmt.Sprintf("Ошибка сохранения в MongoDB: %v", err))
	}

	utils.LogSuccess("Полный анализ готов, отправляем ответ клиенту")
	utils.LogInfo(fmt.Sprintf("Тип документа: %s, длина анализа: %d символов", docType, len(analysis)))

	return gin.H{
		"analysis":      analysis,
		"timestamp":     time.Now().Format(time.RFC3339),
		"document_type": docType,
		"filename":      filename,
	}, nil
}

func AnalyzeText(text string) (string, string, error) {
	parts := utils.SplitText(text, 12000)
	utils.LogInfo(fmt.Sprintf("Документ разбит на %d частей для анализа", len(parts)))

	var analysisResults []string
	for i, part := range parts {
		partNum := i + 1
		utils.LogAction(fmt.Sprintf("Анализ части %d/%d...", partNum, len(parts)))

		result, err := analyzeDocumentPart(part)
		if err != nil {
			utils.LogError(fmt.Sprintf("При анализе части %d: %v", partNum, err))
			return "", "", err
		}

		utils.LogSuccess(fmt.Sprintf("Анализ части %d завершён, результат длиной %d символов", partNum, len(result)))
		analysisResults = append(analysisResults, result)
	}

	fullAnalysis := strings.Join(analysisResults, "\n\n---\n\n")
	docType := detectDocumentType(text)

	return fullAnalysis, docType, nil
}

func analyzeDocumentPart(text string) (string, error) {
	utils.LogInfo(fmt.Sprintf("Отправка запроса к Python AI с текстом длиной %d символов", len(text)))

	result, err := queryPythonAnalysisAPI(text)
	if err != nil {
		return "", err
	}

	utils.LogSuccess(fmt.Sprintf("Успешно получен ответ от AI длиной %d символов", len(result)))
	return result, nil
}

type PythonAnalysisRequest struct {
	Text string `json:"text"`
}

type PythonAnalysisResponse struct {
	Result string `json:"result"`
}

func queryPythonAnalysisAPI(text string) (string, error) {
	payload := PythonAnalysisRequest{
		Text: text,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("ошибка маршалинга payload: %w", err)
	}

	req, err := http.NewRequest("POST", getAnalyzeEndpoint(), bytes.NewBuffer(body))
	if err != nil {
		return "", fmt.Errorf("ошибка создания запроса: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	// Reuse the package-level persistent HTTP client (avoids per-request TCP handshake)
	resp, err := aiHTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ошибка запроса к Python API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ошибка от Python API: статус %d - %s", resp.StatusCode, string(bodyBytes))
	}

	var res PythonAnalysisResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", fmt.Errorf("не удалось распарсить ответ Python API: %w", err)
	}

	return res.Result, nil
}

func GetRelevantLaws() []map[string]string {
	return []map[string]string{
		{"name": "Гражданский кодекс РК", "url": "https://adilet.zan.kz/rus/docs/K950001000_"},
		{"name": "Налоговый кодекс РК", "url": "https://adilet.zan.kz/rus/docs/K2100000409"},
		{"name": "Трудовой кодекс РК", "url": "https://adilet.zan.kz/rus/docs/K1500000011"},
		{"name": "Кодекс об административных правонарушениях РК", "url": "https://adilet.zan.kz/rus/docs/K1400000233"},
	}
}

func GetUserHistory(userID string) ([]map[string]interface{}, error) {
	return repositories.GetUserHistory(userID)
}

func detectDocumentType(text string) string {
	text = strings.ToLower(text)
	switch {
	case strings.Contains(text, "договор"):
		return "Договор"
	case strings.Contains(text, "приказ"):
		return "Приказ"
	case strings.Contains(text, "постановление"):
		return "Постановление"
	case strings.Contains(text, "закон"):
		return "Закон"
	case strings.Contains(text, "решение"):
		return "Решение"
	default:
		return "Неизвестно"
	}
}

var (
	userCache      = make(map[string]string)
	activeAnalysis = make(map[string]context.CancelFunc)
	cacheMutex     sync.Mutex
)

func CancelUserAnalysis(userID string) error {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	if cancel, exists := activeAnalysis[userID]; exists {
		cancel()
		delete(activeAnalysis, userID)
		delete(userCache, userID)
		return nil
	}
	return fmt.Errorf("анализ не найден или уже завершен")
}

func ClearUserCache(userID string) error {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	delete(userCache, userID)
	return nil
}

func CacheUserFile(userID, content string) {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	userCache[userID] = content
}

func GetCachedFile(userID string) (string, bool) {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	content, exists := userCache[userID]
	return content, exists
}

func StartAnalysis(ctx context.Context, userID string, fn func()) context.Context {
	cacheMutex.Lock()
	defer cacheMutex.Unlock()

	ctx, cancel := context.WithCancel(ctx)
	activeAnalysis[userID] = cancel

	go func() {
		fn()
		cacheMutex.Lock()
		delete(activeAnalysis, userID)
		cacheMutex.Unlock()
	}()

	return ctx
}
