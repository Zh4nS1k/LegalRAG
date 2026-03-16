package controllers

import (
	"legally/models"
	"legally/repositories"
	"net/http"

	"github.com/gin-gonic/gin"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

// ReviewerGetMyTasks handles tasks assigned to the current expert
func ReviewerGetMyTasks(c *gin.Context) {
	userID, _ := c.Get("userId")
	tasks, err := repositories.GetEvaluationTasksByUser(userID.(string))
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, tasks)
}

// ReviewerSubmitEvaluation handles expert evaluation submission
func ReviewerSubmitEvaluation(c *gin.Context) {
	var req struct {
		TaskID          string `json:"task_id" binding:"required"`
		AnswerRating    int    `json:"answer_rating"`
		AnswerComment   string `json:"answer_comment"`
		ChunksRating    int    `json:"chunks_rating"`
		ChunksComment   string `json:"chunks_comment"`
		ArticlesRating  int    `json:"articles_rating"`
		ArticlesComment string `json:"articles_comment"`
		ConfirmAction   bool   `json:"confirm_action"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid data format: " + err.Error()})
		return
	}

	taskID, err := primitive.ObjectIDFromHex(req.TaskID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid Task ID"})
		return
	}

	userID, _ := c.Get("userId")

	result := models.EvaluationResult{
		TaskID:          taskID,
		UserID:          userID.(string),
		AnswerRating:    req.AnswerRating,
		AnswerComment:   req.AnswerComment,
		ChunksRating:    req.ChunksRating,
		ChunksComment:   req.ChunksComment,
		ArticlesRating:  req.ArticlesRating,
		ArticlesComment: req.ArticlesComment,
		ConfirmAction:   req.ConfirmAction,
	}

	if err := repositories.SaveEvaluationResult(result); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Evaluation saved successfully"})
}
