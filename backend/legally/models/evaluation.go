package models

import (
	"time"

	"go.mongodb.org/mongo-driver/bson/primitive"
)

type EvaluationStatus string

const (
	StatusPendingAssignment EvaluationStatus = "PENDING_ASSIGNMENT"
	StatusAssigned          EvaluationStatus = "ASSIGNED"
	StatusCompleted         EvaluationStatus = "COMPLETED"
)

type EvaluationTask struct {
	ID               primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	ExternalID       string             `bson:"external_id" json:"external_id"` // ID from Excel/JSON
	Question         string             `bson:"question" json:"question"`
	Answer           string             `bson:"answer" json:"answer"`
	Chunks           []string           `bson:"chunks" json:"chunks"`
	Articles         []string           `bson:"articles" json:"articles"`
	AssignedToUserID string             `bson:"assigned_to_user_id" json:"assigned_to_user_id"`
	Status           EvaluationStatus   `bson:"status" json:"status"`
	Result           *EvaluationResult  `bson:"result,omitempty" json:"result,omitempty"`
	CreatedAt        time.Time          `bson:"created_at" json:"created_at"`
	UpdatedAt        time.Time          `bson:"updated_at" json:"updated_at"`
}

type EvaluationResult struct {
	ID              primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	TaskID          primitive.ObjectID `bson:"task_id" json:"task_id"`
	UserID          string             `bson:"user_id" json:"user_id"`
	AnswerRating    int                `bson:"answer_rating" json:"answer_rating"`
	AnswerComment   string             `bson:"answer_comment" json:"answer_comment"`
	ChunksRating    int                `bson:"chunks_rating" json:"chunks_rating"`
	ChunksComment   string             `bson:"chunks_comment" json:"chunks_comment"`
	ArticlesRating  int                `bson:"articles_rating" json:"articles_rating"`
	ArticlesComment string             `bson:"articles_comment" json:"articles_comment"`
	ConfirmAction   bool               `bson:"confirm_action" json:"confirm_action"` // Keep or Change
	SubmittedAt     time.Time          `bson:"submitted_at" json:"submitted_at"`
}
