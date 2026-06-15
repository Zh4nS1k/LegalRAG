// analysis.go

package models

import "time"

type Analysis struct {
	Filename  string    `bson:"filename" json:"filename"`
	Type      string    `bson:"type" json:"type"`
	Analysis  string    `bson:"analysis" json:"analysis"`
	Text      string    `bson:"text" json:"text"`
	CreatedAt time.Time `bson:"created_at" json:"created_at"`
}
