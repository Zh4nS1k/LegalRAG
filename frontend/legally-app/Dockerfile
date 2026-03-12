# 1. Используем Node.js 20 (как того требуют библиотеки)
FROM node:20-alpine AS builder

WORKDIR /app

# 2. Копируем файлы зависимостей
COPY package*.json ./

# 3. Используем 'npm install' вместо 'npm ci', чтобы сгладить конфликты версий в lock-файле
# Если проект на React, иногда помогает флаг --legacy-peer-deps
RUN npm install

# 4. Копируем остальной код
COPY . .

# 5. Собираем проект
RUN npm run build

# --- Production stage ---
FROM nginx:alpine

# ВНИМАНИЕ: Проверьте, куда ваш билд складывает файлы. 
# В Vite это обычно /app/dist, в Create React App — /app/build.
COPY --from=builder /app/build /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]