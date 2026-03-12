// verify_controller.go — Email OTP verification endpoints

package controllers

import (
	"legally/services"
	"legally/utils"
	"net/http"

	"github.com/gin-gonic/gin"
)

type SendVerificationRequest struct {
	Email string `json:"email" binding:"required,email"`
}

type VerifyCodeRequest struct {
	Email string `json:"email" binding:"required,email"`
	Code  string `json:"code"  binding:"required,len=6"`
}

// SendVerificationCode sends a 6-digit OTP to the given email.
// POST /api/send-verification
func SendVerificationCode(c *gin.Context) {
	var req SendVerificationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Введите корректный email"})
		return
	}

	if err := services.SendVerificationEmail(req.Email); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "Не удалось отправить код. Попробуйте позже.",
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Код подтверждения отправлен на " + req.Email})
}

// VerifyEmailCode validates the OTP and marks the email as verified.
// POST /api/verify-code
func VerifyEmailCode(c *gin.Context) {
	var req VerifyCodeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Введите email и 6-значный код"})
		return
	}

	if err := services.VerifyOTP(req.Email, req.Code); err != nil {
		c.JSON(http.StatusUnauthorized, gin.H{
			"error":      err.Error(),
			"error_code": "INVALID_OTP",
		})
		return
	}

	// Fetch user to generate fresh tokens
	user, err := services.GetUserByEmail(req.Email)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch user"})
		return
	}
	
	accessToken, refreshToken, _ := utils.GenerateTokenPair(user.ID.Hex(), user.Role)

	c.JSON(http.StatusOK, gin.H{
		"message":        "Email успешно подтверждён",
		"email_verified": true,
		"accessToken":    accessToken,
		"refreshToken":   refreshToken,
	})
}
