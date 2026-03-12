// google_auth_controller.go — Google OAuth 2.0 login/register flow

package controllers

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"legally/services"
	"legally/utils"
	"net/http"
	"os"

	"github.com/gin-gonic/gin"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

var googleOAuthConfig *oauth2.Config

// getConfig lazily initializes and returns the OAuth config
// so that godotenv in main.go has time to parse .env first.
func getConfig() *oauth2.Config {
	if googleOAuthConfig == nil {
		googleOAuthConfig = &oauth2.Config{
			ClientID:     os.Getenv("GOOGLE_CLIENT_ID"),
			ClientSecret: os.Getenv("GOOGLE_CLIENT_SECRET"),
			RedirectURL:  os.Getenv("GOOGLE_REDIRECT_URI"),
			Scopes: []string{
				"https://www.googleapis.com/auth/userinfo.email",
				"https://www.googleapis.com/auth/userinfo.profile",
			},
			Endpoint: google.Endpoint,
		}
	}
	return googleOAuthConfig
}

// GoogleLogin redirects the user to Google's OAuth consent screen.
// GET /api/auth/google
func GoogleLogin(c *gin.Context) {
	cfg := getConfig()
	if cfg.ClientID == "" {
		c.JSON(http.StatusServiceUnavailable, gin.H{
			"error": "Google OAuth не настроен. Установите GOOGLE_CLIENT_ID и GOOGLE_CLIENT_SECRET в .env",
		})
		return
	}
	// state = random nonce (in production use a signed CSRF token)
	state := "legally_oauth_state"
	url := cfg.AuthCodeURL(state, oauth2.AccessTypeOffline)
	c.Redirect(http.StatusTemporaryRedirect, url)
}

type googleUserInfo struct {
	ID    string `json:"id"`
	Email string `json:"email"`
	Name  string `json:"name"`
}

// GoogleCallback handles the OAuth callback, upserts the user, and returns JWT tokens.
// GET /api/auth/google/callback
func GoogleCallback(c *gin.Context) {
	code := c.Query("code")
	if code == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Отсутствует код авторизации"})
		return
	}

	cfg := getConfig()
	token, err := cfg.Exchange(context.Background(), code)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Не удалось обменять код: " + err.Error()})
		return
	}

	// Fetch user profile from Google
	resp, err := http.Get(fmt.Sprintf(
		"https://www.googleapis.com/oauth2/v2/userinfo?access_token=%s",
		token.AccessToken,
	))
	if err != nil || resp.StatusCode != http.StatusOK {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Не удалось получить профиль Google"})
		return
	}
	defer resp.Body.Close()

	body, _ := io.ReadAll(resp.Body)
	var profile googleUserInfo
	if err = json.Unmarshal(body, &profile); err != nil || profile.ID == "" {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Невалидный ответ от Google"})
		return
	}

	// Find or create user
	user, err := services.UpsertGoogleUser(profile.ID, profile.Email, profile.Name)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Ошибка сохранения пользователя"})
		return
	}

	accessToken, refreshToken, err := utils.GenerateTokenPair(user.ID.Hex(), user.Role)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Ошибка генерации токена"})
		return
	}

	// Redirect to frontend with tokens in query params (frontend stores them)
	frontendURL := os.Getenv("FRONTEND_URL")
	if frontendURL == "" {
		frontendURL = "http://localhost:3000"
	}
	redirectURL := fmt.Sprintf(
		"%s/auth/callback?accessToken=%s&refreshToken=%s",
		frontendURL, accessToken, refreshToken,
	)
	c.Redirect(http.StatusTemporaryRedirect, redirectURL)
}
