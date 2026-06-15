// analysis_repository.go

package repositories

import (
	"context"
	"fmt"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"legally/db"
	"legally/utils"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/options"
)

func SaveAnalysis(userID, filename, docType, analysis, text string) error {
	utils.LogAction("Сохранение анализа в БД")

	objID, err := primitive.ObjectIDFromHex(userID)
	if err != nil {
		return fmt.Errorf("неверный ID пользователя")
	}

	_, err = db.GetCollection("analyses").InsertOne(context.TODO(), bson.M{
		"user_id":    objID,
		"filename":   filename,
		"type":       docType,
		"analysis":   analysis,
		"text":       text,
		"created_at": time.Now(),
	})

	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка сохранения анализа: %v", err))
	} else {
		utils.LogSuccess("Анализ успешно сохранён в БД")
	}

	return err
}

func GetUserHistory(userID string) ([]map[string]interface{}, error) {
	utils.LogAction(fmt.Sprintf("Получение истории анализов для пользователя %s", userID))

	objID, err := primitive.ObjectIDFromHex(userID)
	if err != nil {
		return nil, fmt.Errorf("неверный ID пользователя")
	}

	coll := db.GetCollection("analyses")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	opts := options.Find().
		SetSort(bson.D{{Key: "created_at", Value: -1}}).
		SetLimit(50)

	cursor, err := coll.Find(ctx, bson.M{"user_id": objID}, opts)
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка получения истории: %v", err))
		return nil, err
	}
	defer cursor.Close(ctx)

	var results []map[string]interface{}
	if err := cursor.All(ctx, &results); err != nil {
		utils.LogError(fmt.Sprintf("Ошибка декодирования истории: %v", err))
		return nil, err
	}

	for _, result := range results {
		delete(result, "_id")
	}

	utils.LogSuccess(fmt.Sprintf("Получено %d записей истории", len(results)))
	return results, nil
}
