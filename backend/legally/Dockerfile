# --- Этап сборки (Builder) ---
FROM golang:1.25-alpine AS builder

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем необходимые системные зависимости
RUN apk add --no-cache git

# Сначала копируем только файлы зависимостей (для эффективного кэширования слоев)
COPY go.mod go.sum ./
RUN go mod download

# Копируем весь исходный код
COPY . .

# Собираем бинарный файл
# -ldflags="-s -w" уменьшает размер файла, удаляя отладочную информацию
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -a -installsuffix cgo -o legally main.go

# --- Финальный этап (Runtime) ---
FROM alpine:latest

WORKDIR /app

# Устанавливаем сертификаты (нужны для HTTPS запросов к API) и часовые пояса
RUN apk --no-cache add ca-certificates tzdata

# Копируем бинарный файл из builder
COPY --from=builder /app/legally .

# Копируем статические файлы (templates, .env и т.д.), если они существуют.
# Чтобы билд не падал, если папка вдруг отсутствует, копируем всё содержимое
# или убедитесь, что папка точно есть в корне проекта.
# COPY --from=builder /app/templates ./templates

EXPOSE 8080

# Запуск приложения
CMD ["./legally"]