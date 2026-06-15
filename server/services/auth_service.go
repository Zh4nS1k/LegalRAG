// auth_service.go

package services

import (
	"context"
	"errors"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"golang.org/x/crypto/bcrypt"
	"legally/db"
	"legally/models"
	"legally/utils"
)

var (
	ErrUserExists         = errors.New("пользователь уже существует")
	ErrUserNotFound       = errors.New("этот email не зарегистрирован")
	ErrInvalidCredentials = errors.New("неверный пароль")
	ErrTokenGeneration    = errors.New("ошибка генерации токена")
	ErrEmailNotVerified   = errors.New("email не подтверждён — проверьте почту")
)

// Register creates a new user with an optional name and requested role.
// After registration the user is NOT yet email_verified — verification is done separately.
func Register(email, password, name string, role models.UserRole) (map[string]string, error) {
	var existingUser models.User
	err := db.GetCollection("users").FindOne(
		context.Background(),
		bson.M{"email": email},
	).Decode(&existingUser)
	if err == nil {
		return nil, ErrUserExists
	}

	hashedPassword, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return nil, err
	}

	now := time.Now()
	user := models.User{
		ID:            primitive.NewObjectID(),
		Email:         email,
		Password:      string(hashedPassword),
		Name:          name,
		Role:          role,
		EmailVerified: false,
		CreatedAt:     now,
		UpdatedAt:     now,
	}

	_, err = db.GetCollection("users").InsertOne(context.Background(), user)
	if err != nil {
		return nil, err
	}

	// Send OTP verification email (non-fatal: registration succeeds even if email fails)
	if sendErr := SendVerificationEmail(email); sendErr != nil {
		// Log but don't block registration
		_ = sendErr
	}

	accessToken, refreshToken, err := utils.GenerateTokenPair(user.ID.Hex(), user.Role)
	if err != nil {
		return nil, ErrTokenGeneration
	}

	return map[string]string{
		"accessToken":  accessToken,
		"refreshToken": refreshToken,
	}, nil
}

// Login authenticates a user. Returns distinct errors for missing email vs wrong password.
func Login(email, password string) (map[string]string, error) {
	var user models.User
	err := db.GetCollection("users").FindOne(
		context.Background(),
		bson.M{"email": email},
	).Decode(&user)

	if err != nil {
		// Email not in DB at all
		return nil, ErrUserNotFound
	}

	// Check password — return a different error so the frontend can highlight the right field
	if err = bcrypt.CompareHashAndPassword([]byte(user.Password), []byte(password)); err != nil {
		return nil, ErrInvalidCredentials
	}

	accessToken, refreshToken, err := utils.GenerateTokenPair(user.ID.Hex(), user.Role)
	if err != nil {
		return nil, ErrTokenGeneration
	}

	return map[string]string{
		"accessToken":  accessToken,
		"refreshToken": refreshToken,
	}, nil
}

// UpsertGoogleUser finds or creates a user from a Google OAuth profile.
func UpsertGoogleUser(googleID, email, name string) (*models.User, error) {
	ctx := context.Background()
	var user models.User

	// Try by googleID first, then by email (accounts can be linked)
	err := db.GetCollection("users").FindOne(ctx, bson.M{"google_id": googleID}).Decode(&user)
	if err != nil {
		// Try by email
		err = db.GetCollection("users").FindOne(ctx, bson.M{"email": email}).Decode(&user)
		if err != nil {
			// New user via Google — create with RoleUser
			now := time.Now()
			user = models.User{
				ID:            primitive.NewObjectID(),
				Email:         email,
				Name:          name,
				GoogleID:      googleID,
				Role:          models.RoleUser,
				EmailVerified: true, // Google email is pre-verified
				VerifiedAt:    &now,
				CreatedAt:     now,
				UpdatedAt:     now,
			}
			_, err = db.GetCollection("users").InsertOne(ctx, user)
			if err != nil {
				return nil, err
			}
			return &user, nil
		}
		// Existing email account — link google ID
		now := time.Now()
		_, _ = db.GetCollection("users").UpdateOne(ctx,
			bson.M{"_id": user.ID},
			bson.M{"$set": bson.M{"google_id": googleID, "updatedAt": now}},
		)
	}
	return &user, nil
}

// GetUserByEmail finds a user by their email address
func GetUserByEmail(email string) (*models.User, error) {
	var user models.User
	err := db.GetCollection("users").FindOne(
		context.Background(),
		bson.M{"email": email},
	).Decode(&user)
	if err != nil {
		return nil, ErrUserNotFound
	}
	return &user, nil
}

// RefreshTokens validates a refresh token and issues a new pair.
func RefreshTokens(refreshToken string) (map[string]string, error) {
	claims, err := utils.ParseRefreshToken(refreshToken)
	if err != nil {
		return nil, ErrInvalidCredentials
	}

	userID, err := primitive.ObjectIDFromHex(claims.UserID)
	if err != nil {
		return nil, ErrUserNotFound
	}

	var user models.User
	err = db.GetCollection("users").FindOne(
		context.Background(),
		bson.M{"_id": userID},
	).Decode(&user)
	if err != nil {
		return nil, ErrUserNotFound
	}

	accessToken, refreshToken, err := utils.GenerateTokenPair(user.ID.Hex(), user.Role)
	if err != nil {
		return nil, ErrTokenGeneration
	}

	return map[string]string{
		"accessToken":  accessToken,
		"refreshToken": refreshToken,
	}, nil
}

// ValidateUser fetches the user by ID for the /api/user endpoint.
func ValidateUser(userID string) (*models.User, error) {
	objID, err := primitive.ObjectIDFromHex(userID)
	if err != nil {
		return nil, err
	}
	var user models.User
	err = db.GetCollection("users").FindOne(
		context.Background(),
		bson.M{"_id": objID},
	).Decode(&user)
	if err != nil {
		return nil, ErrUserNotFound
	}
	return &user, nil
}

func GetAllUsers() ([]models.User, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	cursor, err := db.GetCollection("users").Find(ctx, bson.M{})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)
	var users []models.User
	if err = cursor.All(ctx, &users); err != nil {
		return nil, err
	}
	return users, nil
}

func UpdateUserRole(userID string, newRole models.UserRole) error {
	objID, err := primitive.ObjectIDFromHex(userID)
	if err != nil {
		return err
	}
	_, err = db.GetCollection("users").UpdateOne(
		context.Background(),
		bson.M{"_id": objID},
		bson.M{"$set": bson.M{"role": newRole, "updatedAt": time.Now()}},
	)
	return err
}

func DeleteUser(userID string) error {
	objID, err := primitive.ObjectIDFromHex(userID)
	if err != nil {
		return err
	}
	_, err = db.GetCollection("users").DeleteOne(
		context.Background(),
		bson.M{"_id": objID},
	)
	return err
}
