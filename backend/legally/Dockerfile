# Build stage
FROM golang:1.23-alpine AS builder

WORKDIR /app

# Install git for downloading dependencies
RUN apk add --no-cache git

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy the rest of the source code
COPY . .

# Build the Go application
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o legally main.go

# Final stage
FROM alpine:latest

WORKDIR /app

# Install CA certificates for HTTPS requests
RUN apk --no-cache add ca-certificates tzdata

# Copy the pre-built binary file from the previous stage
COPY --from=builder /app/legally .
COPY --from=builder /app/templates ./templates
# Ensure .env is explicitly passed/mounted, we don't copy local .env

EXPOSE 8080

CMD ["./legally"]
