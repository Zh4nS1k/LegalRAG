package repositories

import (
	"context"
	"legally/db"
	"legally/models"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo/options"
)

func ListEvaluationTasksPaginated(filter bson.M, page, limit int) ([]models.EvaluationTask, int64, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	coll := db.GetCollection(CollQuestionsForReview)
	total, _ := coll.CountDocuments(ctx, filter)

	findOptions := options.Find()
	findOptions.SetSkip(int64((page - 1) * limit))
	findOptions.SetLimit(int64(limit))
	findOptions.SetSort(bson.M{"created_at": -1})

	cursor, err := coll.Find(ctx, filter, findOptions)
	if err != nil {
		return nil, 0, err
	}
	defer cursor.Close(ctx)

	var tasks []models.EvaluationTask
	if err = cursor.All(ctx, &tasks); err != nil {
		return nil, 0, err
	}
	return tasks, total, nil
}

const (
	CollQuestionsForAI     = "questions_for_ai"
	CollQuestionsForReview = "questions_for_review"
	CollRatedQuestions     = "rated_questions"
)

// AddToQuestionsForAI inserts raw questions from batch upload
func AddToQuestionsForAI(batch []models.EvaluationTask) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var docs []interface{}
	for _, q := range batch {
		q.CreatedAt = time.Now()
		q.UpdatedAt = time.Now()
		q.Status = "Parsed"
		docs = append(docs, q)
	}

	_, err := db.GetCollection(CollQuestionsForAI).InsertMany(ctx, docs)
	return err
}

func CreateEvaluationTask(task models.EvaluationTask) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	task.CreatedAt = time.Now()
	task.UpdatedAt = time.Now()
	if task.Status == "" {
		task.Status = "Reviewing"
	}

	_, err := db.GetCollection(CollQuestionsForReview).InsertOne(ctx, task)
	return err
}

func GetAllEvaluationTasks() ([]models.EvaluationTask, error) {
	return ListEvaluationTasks(bson.M{})
}

func ListEvaluationTasks(filter bson.M) ([]models.EvaluationTask, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var tasks []models.EvaluationTask
	cursor, err := db.GetCollection(CollQuestionsForReview).Find(ctx, filter)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	if err = cursor.All(ctx, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

func GetEvaluationTaskByID(id primitive.ObjectID) (models.EvaluationTask, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var task models.EvaluationTask
	err := db.GetCollection(CollQuestionsForReview).FindOne(ctx, bson.M{"_id": id}).Decode(&task)
	return task, err
}

func GetEvaluationTasksByUser(userID string) ([]models.EvaluationTask, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var tasks []models.EvaluationTask
	cursor, err := db.GetCollection(CollQuestionsForReview).Find(ctx, bson.M{"assigned_to_user_id": userID})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	if err = cursor.All(ctx, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

func SaveEvaluationResult(result models.EvaluationResult) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	result.SubmittedAt = time.Now()

	// Update the task with the result and change status to COMPLETED
	_, err := db.GetCollection(CollQuestionsForReview).UpdateOne(
		ctx,
		bson.M{"_id": result.TaskID},
		bson.M{
			"$set": bson.M{
				"result":     result,
				"status":     models.StatusCompleted,
				"updated_at": time.Now(),
			},
		},
	)
	if err != nil {
		return err
	}

	// Also keep saving to rated_questions for historical/legacy reasons if needed
	_, _ = db.GetCollection(CollRatedQuestions).InsertOne(ctx, result)

	return nil
}

func AssignEvaluationTask(taskID primitive.ObjectID, userID string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := db.GetCollection(CollQuestionsForReview).UpdateOne(
		ctx,
		bson.M{"_id": taskID},
		bson.M{
			"$set": bson.M{
				"assigned_to_user_id": userID,
				"status":              models.StatusAssigned,
				"updated_at":          time.Now(),
			},
		},
	)
	return err
}

func UpdateEvaluationTask(id primitive.ObjectID, updates bson.M) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	updates["updated_at"] = time.Now()
	_, err := db.GetCollection(CollQuestionsForReview).UpdateOne(
		ctx,
		bson.M{"_id": id},
		bson.M{"$set": updates},
	)
	return err
}

func DeleteEvaluationTask(id primitive.ObjectID) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := db.GetCollection(CollQuestionsForReview).DeleteOne(ctx, bson.M{"_id": id})
	return err
}

func GetAllEvaluationResults() ([]models.EvaluationResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var results []models.EvaluationResult
	cursor, err := db.GetCollection(CollRatedQuestions).Find(ctx, bson.M{})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	if err = cursor.All(ctx, &results); err != nil {
		return nil, err
	}
	return results, nil
}

func GetQuestionsForAI() ([]models.EvaluationTask, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	var tasks []models.EvaluationTask
	cursor, err := db.GetCollection(CollQuestionsForAI).Find(ctx, bson.M{})
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	if err = cursor.All(ctx, &tasks); err != nil {
		return nil, err
	}
	return tasks, nil
}

func DeleteQuestionFromAI(taskID primitive.ObjectID) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	_, err := db.GetCollection(CollQuestionsForAI).DeleteOne(ctx, bson.M{"_id": taskID})
	return err
}
