// rag_document.go

package models

import (
	"go.mongodb.org/mongo-driver/bson/primitive"
	"time"
)

type RAGDocument struct {
	ID         primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Title      string             `bson:"title" json:"title"`
	Content    string             `bson:"content" json:"content"`
	Category   string             `bson:"category" json:"category"`
	Source     string             `bson:"source" json:"source"`
	Filename   string             `bson:"filename" json:"filename"`
	Embeddings []float64          `bson:"embeddings" json:"embeddings,omitempty"`
	Chunks     []DocumentChunk    `bson:"chunks" json:"chunks,omitempty"`
	Status     string             `bson:"status" json:"status"` // "pending", "processed", "error"
	UploadedBy primitive.ObjectID `bson:"uploaded_by" json:"uploaded_by"`
	CreatedAt  time.Time          `bson:"created_at" json:"created_at"`
	UpdatedAt  time.Time          `bson:"updated_at" json:"updated_at"`
}

type DocumentChunk struct {
	ID         primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Content    string             `bson:"content" json:"content"`
	Embeddings []float64          `bson:"embeddings" json:"embeddings"`
	StartIndex int                `bson:"start_index" json:"start_index"`
	EndIndex   int                `bson:"end_index" json:"end_index"`
}

type RAGUploadRequest struct {
	Title    string `json:"title" binding:"required"`
	Category string `json:"category" binding:"required"`
	Source   string `json:"source"`
}

type RAGSearchRequest struct {
	Query    string `json:"query" binding:"required"`
	Limit    int    `json:"limit"`
	Category string `json:"category"`
}

type RAGSearchResult struct {
	DocumentID   string  `json:"document_id"`
	Title        string  `json:"title"`
	Content      string  `json:"content"`
	Category     string  `json:"category"`
	Source       string  `json:"source"`
	Similarity   float64 `json:"similarity"`
	ChunkContent string  `json:"chunk_content"`
}
