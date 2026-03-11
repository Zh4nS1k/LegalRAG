// routes.go

package api

import (
	"github.com/gin-gonic/gin"
	"legally/api/controllers"
	"legally/api/middleware"
	"legally/db"
	"legally/models"
)

func SetupRoutes(router *gin.Engine) {
	router.Use(middleware.TimingMiddleware())
	router.Use(middleware.CORSMiddleware())
	router.Use(middleware.LoggerMiddleware())

	// Статика и корневая страница
	router.Static("/static", "./public")
	router.GET("/", func(c *gin.Context) {
		c.File("./public/index.html")
	})

	// Health check
	router.GET("/health", func(c *gin.Context) {
		if err := db.Ping(); err != nil {
			c.JSON(503, gin.H{"status": "unhealthy"})
			return
		}
		c.JSON(200, gin.H{"status": "healthy"})
	})

	// Публичные маршруты
	public := router.Group("/api")
	{
		public.POST("/register", controllers.Register)
		public.POST("/login", controllers.Login)
		public.POST("/refresh", controllers.Refresh)
		public.GET("/validate-token", controllers.ValidateToken)
		public.GET("/laws", controllers.GetRelevantLaws)
		public.GET("/stats", controllers.GetSystemStats)
	}

	private := router.Group("/api")
	private.Use(middleware.AuthRequired(models.RoleUser, models.RoleProfessor, models.RoleStudent, models.RoleAdmin))
	{
		private.POST("/analyze", controllers.AnalyzeDocuments)
		private.GET("/history", controllers.GetHistory)
		private.POST("/logout", controllers.Logout)
		private.GET("/user", controllers.GetUser)
		private.POST("/analysis/cancel", controllers.CancelAnalysis)
		private.POST("/cache/clear", controllers.ClearFileCache)
		private.POST("/chat", controllers.HandleChat)
		private.GET("/chat/history", controllers.GetChatHistory)
		private.DELETE("/chat/history", controllers.ClearChatHistory)
		private.GET("/chat/export", controllers.ExportChatHistory)
	}

	// Админские маршруты
	admin := router.Group("/api/admin")
	admin.Use(middleware.AuthRequired(models.RoleAdmin))
	{
		admin.GET("/users", controllers.AdminGetUsers)
		admin.POST("/users", controllers.AdminCreateUser)
		admin.DELETE("/users/:id", controllers.AdminDeleteUser)
		admin.POST("/users/role", controllers.AdminUpdateUserRole)

		// Task Management CRUD
		admin.GET("/tasks", controllers.AdminGetEvaluationTasks)
		admin.POST("/tasks", controllers.AdminCreateTask)
		admin.GET("/tasks/:id", controllers.AdminGetDetailedTask)
		admin.PUT("/tasks/:id", controllers.AdminUpdateTask)
		admin.DELETE("/tasks/:id", controllers.AdminDeleteTask)
		admin.POST("/tasks/assign", controllers.AdminAssignEvaluationTask)
		
		// Upload Pipelines
		admin.POST("/tasks/upload/generate", controllers.AdminUploadGenerate)
		admin.POST("/tasks/upload/ready", controllers.AdminUploadReady)
		
		// Legacy/Utility
		admin.GET("/eval/parsed", controllers.AdminGetParsedQuestions)
		admin.GET("/eval/rated", controllers.AdminGetRatedResults)
		admin.GET("/eval/export", controllers.AdminExportResults)
	}

	// Маршруты рецензента (Professor/Student)
	evalGroup := router.Group("/api/eval")
	evalGroup.Use(middleware.AuthRequired(models.RoleProfessor, models.RoleStudent))
	{
		evalGroup.GET("/my-tasks", controllers.ReviewerGetMyTasks)
		evalGroup.POST("/submit", controllers.ReviewerSubmitEvaluation)
	}
}
