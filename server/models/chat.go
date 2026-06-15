package models

import (
	"go.mongodb.org/mongo-driver/bson/primitive"
	"time"
)

type SourceDetail struct {
	Title string `bson:"title" json:"title"`
	Text  string `bson:"text" json:"text"`
}

type ChatMessage struct {
	ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	UserID    primitive.ObjectID `bson:"user_id" json:"user_id"`
	ChatID    string             `bson:"chat_id" json:"chat_id"`
	Role      string             `bson:"role" json:"role"` // "user" or "assistant"
	Content   string             `bson:"content" json:"content"`
	Sources   []SourceDetail     `bson:"sources,omitempty" json:"sources,omitempty"`
	CreatedAt time.Time          `bson:"created_at" json:"created_at"`
}
