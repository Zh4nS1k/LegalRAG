# Build stage
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy the application source code
COPY . .

# Build the React application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy the built assets from the builder stage to Nginx's default directory
COPY --from=builder /app/build /usr/share/nginx/html

# Copy a custom nginx configuration if needed
# COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80 (to be mapped in docker-compose)
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
