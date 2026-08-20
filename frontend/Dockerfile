FROM node:22-alpine AS build

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# 生产环境 API 走 nginx 反代，BASE_URL 为 /api
ENV VITE_API_URL=/api
RUN npm run build

# ── 运行阶段 ──
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
