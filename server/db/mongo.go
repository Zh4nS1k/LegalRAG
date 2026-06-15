// mongo.go

package db

import (
	"context"
	"log"
	"os"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

var MongoClient *mongo.Client

// dbName reads the database name from the DB_NAME env var (falls back to "legally").
func dbName() string {
	if name := os.Getenv("DB_NAME"); name != "" {
		return name
	}
	return "legally"
}

func InitMongo() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Enable connection pooling (default pool=100, kept alive between requests)
	clientOptions := options.Client().
		ApplyURI(os.Getenv("MONGO_URI")).
		SetMaxPoolSize(20).
		SetMinPoolSize(2).
		SetMaxConnIdleTime(5 * time.Minute)

	client, err := mongo.Connect(ctx, clientOptions)
	if err != nil {
		log.Fatal("❌ Ошибка подключения к MongoDB:", err)
	}

	if err = client.Ping(ctx, nil); err != nil {
		log.Fatal("❌ MongoDB недоступна:", err)
	}

	MongoClient = client
	log.Println("✅ MongoDB подключена")
}

// EnsureIndexes creates compound indexes for all hot query paths.
// Idempotent — safe to call on every startup.
func EnsureIndexes() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	indexes := []struct {
		collection string
		keys       bson.D
		unique     bool
	}{
		// chats: user history queries filter by user_id then sort by created_at
		{"chats", bson.D{{Key: "user_id", Value: 1}, {Key: "created_at", Value: 1}}, false},
		// analyses: user history queries filter by user_id then sort by created_at descending
		{"analyses", bson.D{{Key: "user_id", Value: 1}, {Key: "created_at", Value: -1}}, false},
		// users: login lookups by email must be fast
		{"users", bson.D{{Key: "email", Value: 1}}, true},
		// eval tasks: reviewer sees tasks by assignee
		{"eval_tasks", bson.D{{Key: "assigned_to", Value: 1}, {Key: "status", Value: 1}}, false},
	}

	for _, idx := range indexes {
		coll := GetCollection(idx.collection)
		model := mongo.IndexModel{
			Keys: idx.keys,
			Options: options.Index().
				SetBackground(true).
				SetUnique(idx.unique).
				SetSparse(true),
		}
		_, err := coll.Indexes().CreateOne(ctx, model)
		if err != nil {
			log.Printf("⚠️  Index on %s: %v", idx.collection, err)
		} else {
			log.Printf("✅ Index ensured: %s %v", idx.collection, idx.keys)
		}
	}

	// TTL index: auto-delete expired OTP codes after expires_at
	vcColl := GetCollection("verification_codes")
	ttlModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "expires_at", Value: 1}},
		Options: options.Index().SetExpireAfterSeconds(0).SetBackground(true),
	}
	if _, err := vcColl.Indexes().CreateOne(ctx, ttlModel); err != nil {
		log.Printf("⚠️  TTL index on verification_codes: %v", err)
	} else {
		log.Println("✅ TTL index ensured: verification_codes.expires_at")
	}

	// Sparse unique index on google_id (only for OAuth users)
	_, _ = GetCollection("users").Indexes().CreateOne(ctx, mongo.IndexModel{
		Keys:    bson.D{{Key: "google_id", Value: 1}},
		Options: options.Index().SetUnique(true).SetSparse(true).SetBackground(true),
	})
}

func Ping() error {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return MongoClient.Ping(ctx, nil)
}

func GetCollection(name string) *mongo.Collection {
	return MongoClient.Database(dbName()).Collection(name)
}
