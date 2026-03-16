// admin_controller.go

package controllers

import (
	"legally/models"
	"legally/repositories"
	"legally/services"
	"legally/utils"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

var ragService = services.NewRAGService()

// UploadRAGDocument handles uploading a new document to the RAG system
func UploadRAGDocument(c *gin.Context) {
	utils.LogAction("Получен запрос на загрузку RAG документа")

	// Parse form data
	var req models.RAGUploadRequest
	if err := c.ShouldBind(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":  "Неверные данные запроса",
			"code":   "INVALID_REQUEST",
			"detail": err.Error(),
		})
		return
	}

	// Upload and process document
	doc, err := ragService.UploadRAGDocument(c, req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":  "Ошибка загрузки документа",
			"code":   "UPLOAD_ERROR",
			"detail": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": "Документ успешно загружен и поставлен в очередь на обработку",
		"document": gin.H{
			"id":       doc.ID.Hex(),
			"title":    doc.Title,
			"category": doc.Category,
			"status":   doc.Status,
		},
	})
}

// SearchRAGDocuments handles searching for documents in the RAG system
func SearchRAGDocuments(c *gin.Context) {
	utils.LogAction("Получен запрос на поиск RAG документов")

	var req models.RAGSearchRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":  "Неверные данные запроса",
			"code":   "INVALID_REQUEST",
			"detail": err.Error(),
		})
		return
	}

	if req.Limit <= 0 {
		req.Limit = 10
	}

	// Search documents
	results, err := ragService.SearchRAGDocuments(req.Query, req.Limit, req.Category)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":  "Ошибка поиска документов",
			"code":   "SEARCH_ERROR",
			"detail": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"results": results,
		"count":   len(results),
	})
}

// GetRAGDocuments retrieves RAG documents with pagination
func GetRAGDocuments(c *gin.Context) {
	utils.LogAction("Получен запрос на получение RAG документов")

	// Parse query parameters
	limitStr := c.DefaultQuery("limit", "20")
	offsetStr := c.DefaultQuery("offset", "0")
	category := c.Query("category")

	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		limit = 20
	}

	offset, err := strconv.Atoi(offsetStr)
	if err != nil {
		offset = 0
	}

	// Get documents
	documents, err := ragService.GetRAGDocuments(limit, offset, category)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":  "Ошибка получения документов",
			"code":   "FETCH_ERROR",
			"detail": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success":   true,
		"documents": documents,
		"count":     len(documents),
		"limit":     limit,
		"offset":    offset,
	})
}

// DeleteRAGDocument deletes a RAG document
func DeleteRAGDocument(c *gin.Context) {
	utils.LogAction("Получен запрос на удаление RAG документа")

	docID := c.Param("id")
	if docID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "ID документа не указан",
			"code":  "MISSING_ID",
		})
		return
	}

	// Delete document
	err := ragService.DeleteRAGDocument(docID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":  "Ошибка удаления документа",
			"code":   "DELETE_ERROR",
			"detail": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": "Документ успешно удален",
	})
}

// GetRAGStats returns statistics about RAG documents
func GetRAGStats(c *gin.Context) {
	utils.LogAction("Получен запрос на статистику RAG документов")

	// Get statistics
	stats, err := ragService.GetRAGStats()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":  "Ошибка получения статистики",
			"code":   "STATS_ERROR",
			"detail": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"stats":   stats,
	})
}

// GetRAGCategories returns available document categories
func GetRAGCategories(c *gin.Context) {
	utils.LogAction("Получен запрос на категории RAG документов")

	categories := []string{
		"Гражданское право",
		"Налоговое право",
		"Трудовое право",
		"Административное право",
		"Уголовное право",
		"Семейное право",
		"Земельное право",
		"Экологическое право",
		"Таможенное право",
		"Банковское право",
		"Корпоративное право",
		"Интеллектуальная собственность",
		"Другое",
	}

	c.JSON(http.StatusOK, gin.H{
		"success":    true,
		"categories": categories,
	})
}

// ReprocessRAGDocument reprocesses a document (regenerates embeddings)
func ReprocessRAGDocument(c *gin.Context) {
	utils.LogAction("Получен запрос на переобработку RAG документа")

	docID := c.Param("id")
	if docID == "" {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "ID документа не указан",
			"code":  "MISSING_ID",
		})
		return
	}

	// Convert string ID to ObjectID
	objID, err := primitive.ObjectIDFromHex(docID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "Неверный ID документа",
			"code":  "INVALID_ID",
		})
		return
	}

	// Update status to pending for reprocessing
	err = repositories.UpdateRAGDocument(objID, map[string]interface{}{
		"status": "pending",
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":  "Ошибка обновления статуса",
			"code":   "UPDATE_ERROR",
			"detail": err.Error(),
		})
		return
	}

	// Trigger reprocessing
	go ragService.ProcessDocumentAsync(objID)

	c.JSON(http.StatusOK, gin.H{
		"success": true,
		"message": "Документ поставлен в очередь на переобработку",
	})
}
