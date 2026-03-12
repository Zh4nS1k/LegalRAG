// email_service.go — OTP email sender using SMTP credentials from .env

package services

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"legally/db"
	"legally/models"
	"math/big"
	"net/http"
	"os"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo/options"
)

// generateOTP returns a 6-digit zero-padded random code.
func generateOTP() (string, error) {
	max := big.NewInt(1000000)
	n, err := rand.Int(rand.Reader, max)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%06d", n.Int64()), nil
}

// SendVerificationEmail generates an OTP, stores it in MongoDB, and emails it.
func SendVerificationEmail(email string) error {
	code, err := generateOTP()
	if err != nil {
		return fmt.Errorf("failed to generate OTP: %w", err)
	}

	// Upsert verification code (one active code per email)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	now := time.Now()
	expires := now.Add(15 * time.Minute)

	filter := bson.M{"email": email}
	update := bson.M{"$set": models.VerificationCode{
		Email:     email,
		Code:      code,
		CreatedAt: now,
		ExpiresAt: expires,
	}}
	opts := options.Update().SetUpsert(true)
	_, err = db.GetCollection("verification_codes").UpdateOne(ctx, filter, update, opts)
	if err != nil {
		return fmt.Errorf("failed to store OTP: %w", err)
	}

	// Send email via SMTP
	return sendSMTPEmail(email, code)
}

func sendSMTPEmail(to, code string) error {
	token := os.Getenv("MAILTRAP_TOKEN")
	if token == "" {
		token = "5822d291a664939c214ebe01b1143a6f" // fallback to provided token
	}
	fromEmail := "hello@demomailtrap.co"

	subject := "Ваш код подтверждения — Legally"
	body := fmt.Sprintf(`Здравствуйте!

Ваш код подтверждения email для Legally:

  %s

Код действителен 15 минут. Если вы не запрашивали регистрацию, просто проигнорируйте это письмо.

С уважением,
Команда Legally`, code)

	// Build Mailtrap API request body
	payloadMap := map[string]interface{}{
		"to": []map[string]string{
			{"email": to},
		},
		"from": map[string]string{
			"email": fromEmail,
			"name":  "Legally Support",
		},
		"subject": subject,
		"text":    body,
		"category": "OTP Verification",
	}

	payloadBytes, err := json.Marshal(payloadMap)
	if err != nil {
		return fmt.Errorf("failed to marshal mailtrap payload: %w", err)
	}

	req, err := http.NewRequest("POST", "https://send.api.mailtrap.io/api/send", bytes.NewBuffer(payloadBytes))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send email via mailtrap: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("mailtrap API returned status %d", resp.StatusCode)
	}

	return nil
}

// VerifyOTP checks if the provided code is valid and not expired, then marks the user as verified.
func VerifyOTP(email, code string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var vc models.VerificationCode
	err := db.GetCollection("verification_codes").FindOne(ctx, bson.M{
		"email": email,
		"code":  code,
	}).Decode(&vc)
	if err != nil {
		return fmt.Errorf("неверный или истёкший код подтверждения")
	}

	if time.Now().After(vc.ExpiresAt) {
		return fmt.Errorf("код подтверждения истёк. Запросите новый")
	}

	// Mark user as verified
	now := time.Now()
	_, err = db.GetCollection("users").UpdateOne(ctx,
		bson.M{"email": email},
		bson.M{"$set": bson.M{
			"email_verified": true,
			"verified_at":    now,
			"updatedAt":      now,
		}},
	)
	if err != nil {
		return fmt.Errorf("не удалось обновить статус верификации")
	}

	// Delete used code
	_, _ = db.GetCollection("verification_codes").DeleteOne(ctx, bson.M{"email": email})
	return nil
}
