package repositories

import (
	"context"
	"fmt"
	"legally/db"
	"legally/models"
	"legally/utils"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo/options"
)

func SaveChatMessage(msg models.ChatMessage) error {
	coll := db.GetCollection("chats")
	_, err := coll.InsertOne(context.TODO(), msg)
	if err != nil {
		utils.LogError(fmt.Sprintf("Ошибка сохранения сообщения: %v", err))
		return err
	}
	return nil
}

func GetChatHistory(userID string) ([]models.ChatMessage, error) {
	return GetRecentChatHistory(userID, 0) // 0 = no limit (legacy behaviour)
}

// GetRecentChatHistory returns the last `limit` messages for a user.
// If limit <= 0, returns all messages (preserved for export/clear use cases).
func GetRecentChatHistory(userID string, limit int) ([]models.ChatMessage, error) {
	objID, err := primitive.ObjectIDFromHex(userID)
	if err != nil {
		return nil, fmt.Errorf("неверный ID пользователя")
	}

	coll := db.GetCollection("chats")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Sort oldest-first for chat flow; limit applied server-side
	opts := options.Find().SetSort(bson.D{{Key: "created_at", Value: 1}})
	if limit > 0 {
		// To get the LAST N we sort descending, limit, then reverse in Go
		opts = options.Find().
			SetSort(bson.D{{Key: "created_at", Value: -1}}).
			SetLimit(int64(limit))
	}

	// Build a filter that checks for user_id as either an ObjectID (new way) or string (old way)
	filter := bson.M{
		"$or": []bson.M{
			{"user_id": objID},
			{"user_id": userID},
		},
	}

	cursor, err := coll.Find(ctx, filter, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var messages []models.ChatMessage
	if err := cursor.All(ctx, &messages); err != nil {
		return nil, err
	}

	// If we fetched in descending order (limited), reverse to restore chronological order
	if limit > 0 {
		for i, j := 0, len(messages)-1; i < j; i, j = i+1, j-1 {
			messages[i], messages[j] = messages[j], messages[i]
		}
	}

	return messages, nil
}

func ClearChatHistory(userID string) error {
	objID, err := primitive.ObjectIDFromHex(userID)
	if err != nil {
		return fmt.Errorf("неверный ID пользователя")
	}

	coll := db.GetCollection("chats")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Delete both old (string) and new (ObjectID) format messages
	filter := bson.M{
		"$or": []bson.M{
			{"user_id": objID},
			{"user_id": userID},
		},
	}
	_, err = coll.DeleteMany(ctx, filter)
	return err
}
