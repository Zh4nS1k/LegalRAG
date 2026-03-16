// auth.go

package middleware

import (
	"fmt"
	"github.com/gin-gonic/gin"
	"legally/models"
	"legally/utils"
	"net/http"
	"strings"
	"time"
)

func AuthRequired(allowedRoles ...models.UserRole) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		authHeader := c.GetHeader("Authorization")
		token := ""

		if authHeader != "" {
			// Check "Bearer <token>" format
			tokenParts := strings.Split(authHeader, " ")
			if len(tokenParts) == 2 && tokenParts[0] == "Bearer" {
				token = tokenParts[1]
			}
		}

		// Fallback to query parameter "token" (for window.open downloads)
		if token == "" {
			token = c.Query("token")
		}

		if token == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error": "Требуется авторизация",
				"code":  "MISSING_TOKEN",
			})
			return
		}

		claims, err := utils.ParseToken(token)
		if err != nil {
			utils.LogError(fmt.Sprintf("Ошибка токена: %v", err))
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{
				"error":   "Неверный или истекший токен",
				"code":    "INVALID_OR_EXPIRED_TOKEN",
				"details": err.Error(),
			})
			return
		}

		// Проверка роли
		if len(allowedRoles) > 0 {
			isAllowed := false
			for _, role := range allowedRoles {
				if claims.Role == role {
					isAllowed = true
					break
				}
			}
			if !isAllowed {
				c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
					"error": "Недостаточно прав",
					"code":  "INSUFFICIENT_PERMISSIONS",
				})
				return
			}
		}

		// Сохраняем данные пользователя в контексте
		c.Set("userId", claims.UserID)
		c.Set("userRole", claims.Role)

		RecordMetric(c, "auth_middleware", time.Since(start))
		c.Next()
	}
}
