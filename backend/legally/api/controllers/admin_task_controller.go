package controllers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"legally/models"
	"legally/repositories"
	"legally/utils"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/xuri/excelize/v2"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

// AdminGetEvaluationTasks handles listing tasks with optional status filtering and pagination
func AdminGetEvaluationTasks(c *gin.Context) {
	status := c.Query("status")
	pageStr := c.DefaultQuery("page", "1")
	limitStr := c.DefaultQuery("limit", "10")

	page, _ := strconv.Atoi(pageStr)
	limit, _ := strconv.Atoi(limitStr)
	if page < 1 {
		page = 1
	}
	if limit < 1 {
		limit = 10
	}

	filter := bson.M{}
	if status != "" {
		filter["status"] = status
	}

	tasks, total, err := repositories.ListEvaluationTasksPaginated(filter, page, limit)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"tasks": tasks,
		"total": total,
		"page":  page,
		"limit": limit,
	})
}

// AdminCreateTask handles manual creation of a single task
func AdminCreateTask(c *gin.Context) {
	var task models.EvaluationTask
	if err := c.ShouldBindJSON(&task); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid data format"})
		return
	}

	task.ID = primitive.NewObjectID()
	task.Status = models.StatusPendingAssignment
	if err := repositories.CreateEvaluationTask(task); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, task)
}

// AdminUpdateTask handles manual editing of a task
func AdminUpdateTask(c *gin.Context) {
	idHex := c.Param("id")
	objID, err := primitive.ObjectIDFromHex(idHex)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID format"})
		return
	}

	var updates bson.M
	if err := c.ShouldBindJSON(&updates); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid update data"})
		return
	}

	if err := repositories.UpdateEvaluationTask(objID, updates); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Task updated successfully"})
}

// AdminDeleteTask handles task removal
func AdminDeleteTask(c *gin.Context) {
	idHex := c.Param("id")
	objID, err := primitive.ObjectIDFromHex(idHex)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID format"})
		return
	}

	if err := repositories.DeleteEvaluationTask(objID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Task deleted"})
}

// AdminGetDetailedTask returns a single task with all results
func AdminGetDetailedTask(c *gin.Context) {
	idHex := c.Param("id")
	objID, err := primitive.ObjectIDFromHex(idHex)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid ID format"})
		return
	}

	task, err := repositories.GetEvaluationTaskByID(objID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Task not found"})
		return
	}

	c.JSON(http.StatusOK, task)
}

// AdminUploadGenerate handles Pipeline 1: Raw questions -> AI generation -> DB
func AdminUploadGenerate(c *gin.Context) {
	questions, err := parseUploadFile(c)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// 2. Wrap into tasks for AI pipeline
	var tasks []models.EvaluationTask
	for _, q := range questions {
		tasks = append(tasks, models.EvaluationTask{
			ID:         primitive.NewObjectID(),
			ExternalID: q.ID,
			Question:   q.Question,
			Status:     "Parsed", // Initial status for AI processing
		})
	}

	// Use existing batch processor but ensure it updates status to PENDING_ASSIGNMENT
	go processBatchAndSetPending(tasks)

	c.JSON(http.StatusAccepted, gin.H{"message": fmt.Sprintf("Started AI generation for %d questions", len(tasks))})
}

// AdminUploadReady handles Pipeline 2: Pre-generated tasks -> Direct DB
func AdminUploadReady(c *gin.Context) {
	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No file uploaded"})
		return
	}

	openedFile, _ := file.Open()
	defer openedFile.Close()

	var tasks []models.EvaluationTask
	if strings.HasSuffix(strings.ToLower(file.Filename), ".json") {
		content, _ := io.ReadAll(openedFile)
		if err := json.Unmarshal(content, &tasks); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON format"})
			return
		}
	} else {
		// Excel logic for full tasks
		excelFile, _ := excelize.OpenReader(openedFile)
		rows, _ := excelFile.GetRows(excelFile.GetSheetList()[0])
		for i, row := range rows {
			if i == 0 {
				continue
			} // Header
			if len(row) >= 5 {
				tasks = append(tasks, models.EvaluationTask{
					ExternalID: row[0],
					Question:   row[1],
					Answer:     row[2],
					Chunks:     strings.Split(row[3], "|"), // Assuming pipe separator
					Articles:   strings.Split(row[4], "|"),
				})
			}
		}
	}

	for i := range tasks {
		tasks[i].ID = primitive.NewObjectID()
		tasks[i].Status = models.StatusPendingAssignment
		tasks[i].CreatedAt = time.Now()
		tasks[i].UpdatedAt = time.Now()
		_ = repositories.CreateEvaluationTask(tasks[i])
	}

	c.JSON(http.StatusOK, gin.H{"message": fmt.Sprintf("Successfully imported %d ready tasks", len(tasks))})
}

// Helper to parse upload (reused from evaluation_controller)
func parseUploadFile(c *gin.Context) ([]BatchQuestion, error) {
	file, err := c.FormFile("file")
	if err != nil {
		return nil, fmt.Errorf("file not found")
	}
	openedFile, _ := file.Open()
	defer openedFile.Close()

	var questions []BatchQuestion
	if strings.HasSuffix(strings.ToLower(file.Filename), ".json") {
		content, _ := io.ReadAll(openedFile)
		json.Unmarshal(content, &questions)
	} else {
		excelFile, _ := excelize.OpenReader(openedFile)
		rows, _ := excelFile.GetRows(excelFile.GetSheetList()[0])
		for i, row := range rows {
			if i == 0 {
				continue
			}
			if len(row) >= 2 {
				questions = append(questions, BatchQuestion{ID: row[0], Question: row[1]})
			}
		}
	}
	return questions, nil
}

func processBatchAndSetPending(tasks []models.EvaluationTask) {
	pythonAPI := utils.GetAIServiceBaseURL() + "/api/v1/generate-eval-data"
	for _, task := range tasks {
		evalData, err := fetchEvalDataFromPython(pythonAPI, task.Question)
		if err != nil {
			continue
		}

		task.Answer = evalData.Answer
		task.Chunks = evalData.Chunks
		task.Articles = evalData.Articles
		task.Status = models.StatusPendingAssignment
		task.CreatedAt = time.Now()
		task.UpdatedAt = time.Now()

		_ = repositories.CreateEvaluationTask(task)
	}
}

// AdminAssignEvaluationTask handles task assignment to experts
func AdminAssignEvaluationTask(c *gin.Context) {
	var req struct {
		TaskID string `json:"task_id" binding:"required"`
		UserID string `json:"user_id" binding:"required"`
	}
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid data format"})
		return
	}

	objID, _ := primitive.ObjectIDFromHex(req.TaskID)
	if err := repositories.AssignEvaluationTask(objID, req.UserID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Task assigned successfully"})
}

// AdminGetParsedQuestions returns questions waiting for AI generation
func AdminGetParsedQuestions(c *gin.Context) {
	tasks, err := repositories.GetQuestionsForAI()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, tasks)
}

// AdminGetRatedResults returns all legacy rated results
func AdminGetRatedResults(c *gin.Context) {
	results, err := repositories.GetAllEvaluationResults()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, results)
}

// AdminExportResults handles data export in various formats
func AdminExportResults(c *gin.Context) {
	format := c.DefaultQuery("format", "csv")
	results, err := repositories.GetAllEvaluationResults()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if format == "json" {
		c.JSON(http.StatusOK, results)
		return
	}

	// Simple CSV export for now
	c.Header("Content-Type", "text/csv")
	c.Header("Content-Disposition", "attachment;filename=results.csv")

	csv := "TaskID,Question,Rating,Decision\n"
	for _, res := range results {
		decision := "CHANGE"
		if res.ConfirmAction {
			decision = "KEEP"
		}
		csv += fmt.Sprintf("%s,%s,%d,%s\n", res.TaskID.Hex(), "...", res.AnswerRating, decision)
	}
	c.String(http.StatusOK, csv)
}

type BatchQuestion struct {
	ID       string `json:"id"`
	Question string `json:"question"`
}

type PythonEvalDataResponse struct {
	Answer   string   `json:"answer"`
	Chunks   []string `json:"chunks"`
	Articles []string `json:"articles"`
}

func fetchEvalDataFromPython(apiURL, query string) (*PythonEvalDataResponse, error) {
	payload := map[string]string{"query": query}
	body, _ := json.Marshal(payload)

	resp, err := http.Post(apiURL, "application/json", bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Python API returned status %d", resp.StatusCode)
	}

	var data PythonEvalDataResponse
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, err
	}
	return &data, nil
}
