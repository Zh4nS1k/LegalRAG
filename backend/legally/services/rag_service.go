// rag_service.go

package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"legally/models"
	"legally/repositories"
	"legally/utils"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

const (
	embeddingModel = "text-embedding-ada-002"
	embeddingAPI   = "https://api.openai.com/v1/embeddings"
)

type RAGService struct{}

func NewRAGService() *RAGService {
	return &RAGService{}
}

// UploadRAGDocument processes and stores a new document in the RAG system
func (s *RAGService) UploadRAGDocument(c *gin.Context, req models.RAGUploadRequest) (*models.RAGDocument, error) {
	utils.LogAction("Загрузка нового RAG документа")

	// Get file from request
	file, err := c.FormFile("document")
	if err != nil {
		return nil, fmt.Errorf("файл не найден: %w", err)
	}

	// Validate file type
	if !strings.HasSuffix(strings.ToLower(file.Filename), ".pdf") {
		return nil, fmt.Errorf("поддерживаются только PDF файлы")
	}

	// Extract text from PDF
	text, filename, err := utils.ProcessUploadedFile(c)
	if err != nil {
		return nil, fmt.Errorf("ошибка обработки файла: %w", err)
	}

	// Get user ID from context
	userID, exists := c.Get("userId")
	if !exists {
		return nil, fmt.Errorf("пользователь не найден в контексте")
	}

	// Convert userID string to ObjectID
	userObjID, err := primitive.ObjectIDFromHex(userID.(string))
	if err != nil {
		return nil, fmt.Errorf("неверный ID пользователя: %w", err)
	}

	// Create RAG document
	doc := &models.RAGDocument{
		Title:      req.Title,
		Content:    text,
		Category:   req.Category,
		Source:     req.Source,
		Filename:   filename,
		Status:     "pending",
		UploadedBy: userObjID,
	}

	// Save document to database
	err = repositories.SaveRAGDocument(doc)
	if err != nil {
		return nil, fmt.Errorf("ошибка сохранения документа: %w", err)
	}

	// Process document asynchronously (generate embeddings, chunk content)
	go s.ProcessDocumentAsync(doc.ID)

	utils.LogSuccess(fmt.Sprintf("RAG документ успешно загружен: %s", doc.Title))
	return doc, nil
}

// ProcessDocumentAsync processes the document in the background
func (s *RAGService) ProcessDocumentAsync(docID primitive.ObjectID) {
	utils.LogAction(fmt.Sprintf("Асинхронная обработка документа: %s", docID.Hex()))

	// Get document from database
	doc, err := repositories.GetRAGDocument(docID)
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка получения документа: %v", err))
		return
	}

	// Update status to processing
	err = repositories.UpdateRAGDocument(docID, map[string]interface{}{
		"status": "processing",
	})
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка обновления статуса: %v", err))
		return
	}

	// Generate embeddings for the document
	embeddings, err := s.generateEmbeddings(doc.Content)
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка генерации эмбеддингов: %v", err))
		repositories.UpdateRAGDocument(docID, map[string]interface{}{
			"status": "error",
		})
		return
	}

	// Chunk the document
	chunks := s.chunkDocument(doc.Content)

	// Generate embeddings for chunks
	for i := range chunks {
		chunkEmbeddings, err := s.generateEmbeddings(chunks[i].Content)
		if err != nil {
			utils.LogWarning(fmt.Sprintf("Ошибка генерации эмбеддингов для чанка %d: %v", i, err))
			continue
		}
		chunks[i].Embeddings = chunkEmbeddings
	}

	// Update document with embeddings and chunks
	err = repositories.UpdateRAGDocument(docID, map[string]interface{}{
		"embeddings": embeddings,
		"chunks":     chunks,
		"status":     "processed",
	})
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка обновления документа: %v", err))
		return
	}

	utils.LogSuccess(fmt.Sprintf("Документ успешно обработан: %s", doc.Title))
}

// generateEmbeddings generates embeddings for text using OpenAI API
func (s *RAGService) generateEmbeddings(text string) ([]float64, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		// Fallback to simple embedding simulation for development
		return s.generateSimpleEmbeddings(text), nil
	}

	payload := map[string]interface{}{
		"model": embeddingModel,
		"input": text,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("ошибка маршалинга payload: %w", err)
	}

	req, err := http.NewRequest("POST", embeddingAPI, bytes.NewBuffer(body))
	if err != nil {
		return nil, fmt.Errorf("ошибка создания запроса: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ошибка запроса к OpenAI: %w", err)
	}
	defer resp.Body.Close()

	resBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("ошибка чтения ответа: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ошибка от OpenAI API: %s", string(resBody))
	}

	var result struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}

	if err := json.Unmarshal(resBody, &result); err != nil {
		return nil, fmt.Errorf("ошибка парсинга ответа: %w", err)
	}

	if len(result.Data) == 0 {
		return nil, fmt.Errorf("пустой ответ от OpenAI API")
	}

	return result.Data[0].Embedding, nil
}

// generateSimpleEmbeddings generates simple embeddings for development
func (s *RAGService) generateSimpleEmbeddings(text string) []float64 {
	// Simple embedding simulation - in production, use real embeddings
	// This creates a 384-dimensional vector based on character frequency
	embedding := make([]float64, 384)

	for i, char := range text {
		if i >= 384 {
			break
		}
		embedding[i] = float64(char) / 255.0
	}

	return embedding
}

// chunkDocument splits the document into smaller chunks
func (s *RAGService) chunkDocument(content string) []models.DocumentChunk {
	utils.LogAction("Разделение документа на чанки")

	// Simple chunking strategy - split by paragraphs
	paragraphs := strings.Split(content, "\n\n")
	var chunks []models.DocumentChunk

	startIndex := 0
	for _, paragraph := range paragraphs {
		paragraph = strings.TrimSpace(paragraph)
		if len(paragraph) == 0 {
			continue
		}

		// Create chunk
		chunk := models.DocumentChunk{
			ID:         primitive.NewObjectID(),
			Content:    paragraph,
			StartIndex: startIndex,
			EndIndex:   startIndex + len(paragraph),
		}

		chunks = append(chunks, chunk)
		startIndex += len(paragraph) + 2 // +2 for "\n\n"
	}

	utils.LogInfo(fmt.Sprintf("Документ разделен на %d чанков", len(chunks)))
	return chunks
}

// SearchRAGDocuments searches for relevant documents
func (s *RAGService) SearchRAGDocuments(query string, limit int, category string) ([]models.RAGSearchResult, error) {
	utils.LogAction(fmt.Sprintf("Поиск RAG документов: %s", query))

	if limit <= 0 {
		limit = 10
	}

	results, err := repositories.SearchRAGDocuments(query, limit, category)
	if err != nil {
		return nil, fmt.Errorf("ошибка поиска документов: %w", err)
	}

	utils.LogSuccess(fmt.Sprintf("Найдено %d документов", len(results)))
	return results, nil
}

// GetRAGDocuments retrieves RAG documents with pagination
func (s *RAGService) GetRAGDocuments(limit, offset int, category string) ([]models.RAGDocument, error) {
	utils.LogAction("Получение RAG документов")

	if limit <= 0 {
		limit = 20
	}

	var documents []models.RAGDocument
	var err error

	if category != "" {
		documents, err = repositories.GetRAGDocumentsByCategory(category, limit, offset)
	} else {
		documents, err = repositories.GetAllRAGDocuments(limit, offset)
	}

	if err != nil {
		return nil, fmt.Errorf("ошибка получения документов: %w", err)
	}

	utils.LogSuccess(fmt.Sprintf("Получено %d документов", len(documents)))
	return documents, nil
}

// DeleteRAGDocument deletes a RAG document
func (s *RAGService) DeleteRAGDocument(docID string) error {
	utils.LogAction(fmt.Sprintf("Удаление RAG документа: %s", docID))

	objID, err := primitive.ObjectIDFromHex(docID)
	if err != nil {
		return fmt.Errorf("неверный ID документа: %w", err)
	}

	err = repositories.DeleteRAGDocument(objID)
	if err != nil {
		return fmt.Errorf("ошибка удаления документа: %w", err)
	}

	utils.LogSuccess("RAG документ успешно удален")
	return nil
}

// GetRAGStats returns statistics about RAG documents
func (s *RAGService) GetRAGStats() (map[string]interface{}, error) {
	utils.LogAction("Получение статистики RAG документов")

	stats, err := repositories.GetRAGDocumentStats()
	if err != nil {
		return nil, fmt.Errorf("ошибка получения статистики: %w", err)
	}

	utils.LogSuccess("Статистика RAG документов получена")
	return stats, nil
}
