// user.go

package models

import (
	"errors"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"time"
)

type UserRole string

var (
	ErrUserExists         = errors.New("пользователь с таким email уже существует")
	ErrInvalidCredentials = errors.New("неверные учетные данные")
)

const (
	RoleAdmin     UserRole = "admin"
	RoleUser      UserRole = "user"
	RoleStudent   UserRole = "student"
	RoleProfessor UserRole = "professor"
	RoleAnonymous UserRole = "anonymous"
)

type User struct {
	ID            primitive.ObjectID `bson:"_id,omitempty"         json:"id"`
	Email         string             `bson:"email"                 json:"email"`
	Password      string             `bson:"password"              json:"-"`
	Name          string             `bson:"name,omitempty"        json:"name"`
	Role          UserRole           `bson:"role"                  json:"role"`
	EmailVerified bool               `bson:"email_verified"        json:"email_verified"`
	VerifiedAt    *time.Time         `bson:"verified_at,omitempty" json:"verified_at,omitempty"`
	// GoogleID is set when the account was created via Google OAuth
	GoogleID  string    `bson:"google_id,omitempty"  json:"google_id,omitempty"`
	CreatedAt time.Time `bson:"createdAt"            json:"created_at"`
	UpdatedAt time.Time `bson:"updatedAt"            json:"updated_at"`
}

// VerificationCode stored in MongoDB 'verification_codes' collection with TTL index.
type VerificationCode struct {
	ID        primitive.ObjectID `bson:"_id,omitempty"`
	Email     string             `bson:"email"`
	Code      string             `bson:"code"`
	CreatedAt time.Time          `bson:"created_at"`
	ExpiresAt time.Time          `bson:"expires_at"` // TTL index on this field
}

