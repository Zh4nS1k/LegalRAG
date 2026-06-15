// rag_repository.go

package repositories

import (
	"context"
	"fmt"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo/options"
	"legally/db"
	"legally/models"
	"legally/utils"
	"time"
)

func SaveRAGDocument(doc *models.RAGDocument) error {
	utils.LogAction("Сохранение RAG документа в БД")

	if doc.ID.IsZero() {
		doc.ID = primitive.NewObjectID()
	}

	doc.CreatedAt = time.Now()
	doc.UpdatedAt = time.Now()

	_, err := db.GetCollection("rag_documents").InsertOne(context.TODO(), doc)
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка сохранения RAG документа: %v", err))
		return err
	}

	utils.LogSuccess(fmt.Sprintf("RAG документ успешно сохранён: %s", doc.Title))
	return nil
}

func UpdateRAGDocument(id primitive.ObjectID, updates bson.M) error {
	utils.LogAction(fmt.Sprintf("Обновление RAG документа: %s", id.Hex()))

	updates["updated_at"] = time.Now()

	_, err := db.GetCollection("rag_documents").UpdateOne(
		context.TODO(),
		bson.M{"_id": id},
		bson.M{"$set": updates},
	)

	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка обновления RAG документа: %v", err))
		return err
	}

	utils.LogSuccess("RAG документ успешно обновлён")
	return nil
}

func GetRAGDocument(id primitive.ObjectID) (*models.RAGDocument, error) {
	var doc models.RAGDocument
	err := db.GetCollection("rag_documents").FindOne(
		context.TODO(),
		bson.M{"_id": id},
	).Decode(&doc)

	if err != nil {
		return nil, err
	}

	return &doc, nil
}

func GetAllRAGDocuments(limit, offset int) ([]models.RAGDocument, error) {
	utils.LogAction("Получение всех RAG документов")

	opts := options.Find().
		SetSort(bson.D{{Key: "created_at", Value: -1}}).
		SetLimit(int64(limit)).
		SetSkip(int64(offset))

	cursor, err := db.GetCollection("rag_documents").Find(context.TODO(), bson.M{}, opts)
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка получения RAG документов: %v", err))
		return nil, err
	}
	defer cursor.Close(context.TODO())

	var documents []models.RAGDocument
	if err := cursor.All(context.TODO(), &documents); err != nil {
		utils.LogError(fmt.Sprintf("Ошибка декодирования RAG документов: %v", err))
		return nil, err
	}

	utils.LogSuccess(fmt.Sprintf("Получено %d RAG документов", len(documents)))
	return documents, nil
}

func GetRAGDocumentsByCategory(category string, limit, offset int) ([]models.RAGDocument, error) {
	utils.LogAction(fmt.Sprintf("Получение RAG документов категории: %s", category))

	opts := options.Find().
		SetSort(bson.D{{Key: "created_at", Value: -1}}).
		SetLimit(int64(limit)).
		SetSkip(int64(offset))

	filter := bson.M{"category": category}
	cursor, err := db.GetCollection("rag_documents").Find(context.TODO(), filter, opts)
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка получения RAG документов по категории: %v", err))
		return nil, err
	}
	defer cursor.Close(context.TODO())

	var documents []models.RAGDocument
	if err := cursor.All(context.TODO(), &documents); err != nil {
		utils.LogError(fmt.Sprintf("Ошибка декодирования RAG документов: %v", err))
		return nil, err
	}

	utils.LogSuccess(fmt.Sprintf("Получено %d RAG документов категории %s", len(documents), category))
	return documents, nil
}

func DeleteRAGDocument(id primitive.ObjectID) error {
	utils.LogAction(fmt.Sprintf("Удаление RAG документа: %s", id.Hex()))

	_, err := db.GetCollection("rag_documents").DeleteOne(context.TODO(), bson.M{"_id": id})
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка удаления RAG документа: %v", err))
		return err
	}

	utils.LogSuccess("RAG документ успешно удалён")
	return nil
}

func SearchRAGDocuments(query string, limit int, category string) ([]models.RAGSearchResult, error) {
	utils.LogAction(fmt.Sprintf("Поиск RAG документов: %s", query))

	// Simple text search for now - in a real implementation, you'd use vector similarity search
	filter := bson.M{
		"$or": []bson.M{
			{"title": bson.M{"$regex": query, "$options": "i"}},
			{"content": bson.M{"$regex": query, "$options": "i"}},
		},
	}

	if category != "" {
		filter["category"] = category
	}

	opts := options.Find().
		SetSort(bson.D{{Key: "created_at", Value: -1}}).
		SetLimit(int64(limit))

	cursor, err := db.GetCollection("rag_documents").Find(context.TODO(), filter, opts)
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка поиска RAG документов: %v", err))
		return nil, err
	}
	defer cursor.Close(context.TODO())

	var documents []models.RAGDocument
	if err := cursor.All(context.TODO(), &documents); err != nil {
		utils.LogError(fmt.Sprintf("Ошибка декодирования результатов поиска: %v", err))
		return nil, err
	}

	var results []models.RAGSearchResult
	for _, doc := range documents {
		// Simple similarity score based on text matching
		similarity := calculateSimpleSimilarity(query, doc.Title, doc.Content)

		result := models.RAGSearchResult{
			DocumentID:   doc.ID.Hex(),
			Title:        doc.Title,
			Content:      doc.Content,
			Category:     doc.Category,
			Source:       doc.Source,
			Similarity:   similarity,
			ChunkContent: extractRelevantChunk(query, doc.Content),
		}
		results = append(results, result)
	}

	utils.LogSuccess(fmt.Sprintf("Найдено %d RAG документов", len(results)))
	return results, nil
}

func GetRAGDocumentStats() (map[string]interface{}, error) {
	utils.LogAction("Получение статистики RAG документов")

	// Total documents
	total, err := db.GetCollection("rag_documents").CountDocuments(context.TODO(), bson.M{})
	if err != nil {
		return nil, err
	}

	// Documents by status
	pipeline := []bson.M{
		{"$group": bson.M{
			"_id":   "$status",
			"count": bson.M{"$sum": 1},
		}},
	}

	cursor, err := db.GetCollection("rag_documents").Aggregate(context.TODO(), pipeline)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(context.TODO())

	var statusStats []bson.M
	if err := cursor.All(context.TODO(), &statusStats); err != nil {
		return nil, err
	}

	// Documents by category
	pipeline = []bson.M{
		{"$group": bson.M{
			"_id":   "$category",
			"count": bson.M{"$sum": 1},
		}},
	}

	cursor, err = db.GetCollection("rag_documents").Aggregate(context.TODO(), pipeline)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(context.TODO())

	var categoryStats []bson.M
	if err := cursor.All(context.TODO(), &categoryStats); err != nil {
		return nil, err
	}

	stats := map[string]interface{}{
		"total":          total,
		"status_stats":   statusStats,
		"category_stats": categoryStats,
	}

	return stats, nil
}

// Helper functions
func calculateSimpleSimilarity(query, title, content string) float64 {
	// Simple similarity calculation based on text matching
	// In a real implementation, you'd use vector similarity
	queryLower := utils.ToLower(query)
	titleLower := utils.ToLower(title)
	contentLower := utils.ToLower(content)

	titleMatch := 0.0
	contentMatch := 0.0

	// Check title similarity
	if utils.Contains(titleLower, queryLower) {
		titleMatch = 0.8
	}

	// Check content similarity
	if utils.Contains(contentLower, queryLower) {
		contentMatch = 0.6
	}

	return titleMatch + contentMatch
}

func extractRelevantChunk(query, content string) string {
	// Extract a relevant chunk of content around the query
	// This is a simplified version - in a real implementation, you'd use proper chunking
	if len(content) <= 200 {
		return content
	}

	queryLower := utils.ToLower(query)
	contentLower := utils.ToLower(content)

	index := utils.IndexOf(contentLower, queryLower)
	if index == -1 {
		// If query not found, return first 200 characters
		return content[:200] + "..."
	}

	start := index - 100
	if start < 0 {
		start = 0
	}

	end := index + len(query) + 100
	if end > len(content) {
		end = len(content)
	}

	return content[start:end]
}
