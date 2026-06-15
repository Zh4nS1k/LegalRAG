// jwt.go

package utils

import (
	"github.com/golang-jwt/jwt/v5"
	"legally/models"
	"os"
	"time"
)

var (
	jwtSecret        = []byte(os.Getenv("JWT_SECRET"))
	jwtRefreshSecret = []byte(os.Getenv("JWT_REFRESH_SECRET"))
)

type Claims struct {
	UserID string          `json:"userId"`
	Role   models.UserRole `json:"role"`
	jwt.RegisteredClaims
}

func GenerateTokenPair(userID string, role models.UserRole) (string, string, error) {
	// Access Token (1 час	)
	accessToken, err := generateToken(userID, role, 1*time.Hour, jwtSecret)
	if err != nil {
		return "", "", err
	}

	// Refresh Token (7 дней)
	refreshToken, err := generateToken(userID, role, 168*time.Hour, jwtRefreshSecret)

	return accessToken, refreshToken, err
}

func generateToken(userID string, role models.UserRole, duration time.Duration, secret []byte) (string, error) {
	claims := &Claims{
		UserID: userID,
		Role:   role,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(duration)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(secret)
}

func ParseToken(tokenString string) (*Claims, error) {
	return parseToken(tokenString, jwtSecret)
}

func ParseRefreshToken(tokenString string) (*Claims, error) {
	return parseToken(tokenString, jwtRefreshSecret)
}

func parseToken(tokenString string, secret []byte) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		return secret, nil
	})

	if err != nil {
		return nil, err
	}

	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return claims, nil
	}

	return nil, jwt.ErrInvalidKey
}
