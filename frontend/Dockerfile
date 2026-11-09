# ============================================================
# Smart ICU Frontend - Multi-Stage Dockerfile
# Stage 1: Build static assets using locked npm dependencies
# ============================================================
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package manifests and install dependencies reproducibly via npm ci
COPY package.json package-lock.json ./
RUN npm ci

# Copy source code and build config
COPY . .

# Build production bundle (tsc -b && vite build)
RUN npm run build

# ============================================================
# Stage 2: Lightweight Nginx production web server
# ============================================================
FROM nginx:1.27-alpine AS runner

# Remove default Nginx welcome page
RUN rm -rf /usr/share/nginx/html/*

# Copy compiled SPA bundle from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom reverse-proxy and SPA routing configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose HTTP port 80
EXPOSE 80

# Health check verifying web server response
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:80/ || exit 1

# Start Nginx in foreground
CMD ["nginx", "-g", "daemon off;"]
