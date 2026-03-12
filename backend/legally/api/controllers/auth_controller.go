// auth_controller.go

package controllers

import (
	"errors"
	"legally/models"
	"legally/services"
	"net/http"

	"github.com/gin-gonic/gin"
)

// RegisterRequest allows specifying an optional name and required role.
type RegisterRequest struct {
	Email    string          `json:"email"    binding:"required,email"`
	Password string          `json:"password" binding:"required,min=8"`
	Name     string          `json:"name"`                                    // optional
	Role     models.UserRole `json:"role"     binding:"required,oneof=user student professor"`
}

type RefreshRequest struct {
	RefreshToken string `json:"refreshToken" binding:"required"`
}

func Register(c *gin.Context) {
	var req RegisterRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":      "Проверьте введённые данные: " + bindingErrorMessage(err),
			"error_code": "VALIDATION_ERROR",
		})
		return
	}

	tokens, err := services.Register(req.Email, req.Password, req.Name, req.Role)
	if err != nil {
		status := http.StatusBadRequest
		code := "REGISTER_ERROR"
		if errors.Is(err, services.ErrUserExists) {
			status = http.StatusConflict
			code = "EMAIL_EXISTS"
		}
		c.JSON(status, gin.H{"error": err.Error(), "error_code": code})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":        "Регистрация прошла успешно. Мы отправили код подтверждения на ваш email.",
		"accessToken":    tokens["accessToken"],
		"refreshToken":   tokens["refreshToken"],
		"email_verified": false,
	})
}

func Login(c *gin.Context) {
	var req struct {
		Email    string `json:"email"    binding:"required,email"`
		Password string `json:"password" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":      "Введите email и пароль",
			"error_code": "VALIDATION_ERROR",
		})
		return
	}

	tokens, err := services.Login(req.Email, req.Password)
	if err != nil {
		// Return distinct error_code so the frontend can highlight the right field
		code := "LOGIN_ERROR"
		status := http.StatusUnauthorized
		if errors.Is(err, services.ErrUserNotFound) {
			code = "EMAIL_NOT_FOUND"
			status = http.StatusNotFound
		} else if errors.Is(err, services.ErrInvalidCredentials) {
			code = "WRONG_PASSWORD"
		}
		c.JSON(status, gin.H{
			"error":      err.Error(),
			"error_code": code,
			"success":    false,
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message":      "Вход выполнен успешно",
		"accessToken":  tokens["accessToken"],
		"refreshToken": tokens["refreshToken"],
		"success":      true,
	})
}

func GetUser(c *gin.Context) {
	userID, exists := c.Get("userId")
	if !exists {
		c.JSON(http.StatusUnauthorized, gin.H{"error": "Authentication required"})
		return
	}

	user, err := services.ValidateUser(userID.(string))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Error fetching user data"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"email":          user.Email,
		"name":           user.Name,
		"role":           user.Role,
		"email_verified": user.EmailVerified,
		"createdAt":      user.CreatedAt,
	})
}

func Refresh(c *gin.Context) {
	var req RefreshRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	tokens, err := services.RefreshTokens(req.RefreshToken)
	if err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{"error": err.Error(), "success": false})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"accessToken":  tokens["accessToken"],
		"refreshToken": tokens["refreshToken"],
		"success":      true,
	})
}

func ValidateToken(c *gin.Context) {
	token := c.GetHeader("Authorization")
	if token == "" {
		c.JSON(http.StatusUnauthorized, gin.H{"valid": false})
		return
	}
	c.JSON(http.StatusOK, gin.H{"valid": true, "message": "Токен действителен"})
}

func Logout(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"message": "Выход выполнен успешно", "success": true})
}

// ─── Admin ────────────────────────────────────────────────────────────────────

func AdminGetUsers(c *gin.Context) {
	users, err := services.GetAllUsers()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch users"})
		return
	}

	type UserInfo struct {
		ID    string          `json:"id"`
		Email string          `json:"email"`
		Name  string          `json:"name"`
		Role  models.UserRole `json:"role"`
	}

	var response []UserInfo
	for _, u := range users {
		response = append(response, UserInfo{
			ID:    u.ID.Hex(),
			Email: u.Email,
			Name:  u.Name,
			Role:  u.Role,
		})
	}
	c.JSON(http.StatusOK, response)
}

func AdminUpdateUserRole(c *gin.Context) {
	var req struct {
		UserID string          `json:"user_id" binding:"required"`
		Role   models.UserRole `json:"role"    binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := services.UpdateUserRole(req.UserID, req.Role); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update role"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "Role updated successfully"})
}

func AdminCreateUser(c *gin.Context) {
	var req struct {
		Email    string          `json:"email"    binding:"required,email"`
		Password string          `json:"password" binding:"required,min=8"`
		Name     string          `json:"name"`
		Role     models.UserRole `json:"role"     binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if _, err := services.Register(req.Email, req.Password, req.Name, req.Role); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create user: " + err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "User created successfully"})
}

func AdminDeleteUser(c *gin.Context) {
	userID := c.Param("id")
	if userID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "User ID is required"})
		return
	}
	if err := services.DeleteUser(userID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to delete user"})
		return
	}
	c.JSON(http.StatusOK, gin.H{"message": "User deleted successfully"})
}

// bindingErrorMessage converts a gin/validator error into a user-friendly Russian message.
func bindingErrorMessage(err error) string {
	msg := err.Error()
	if len(msg) > 120 {
		return "некорректные данные"
	}
	return msg
}
