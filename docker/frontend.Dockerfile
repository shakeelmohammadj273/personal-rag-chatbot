FROM node:20-alpine AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
# The page calls /api/..., and nginx forwards that to the backend (no CORS needed)
ARG VITE_API_URL=/api
ENV VITE_API_URL=$VITE_API_URL
RUN npm run build

FROM nginx:alpine
# nginx config is inline, so this build needs no extra file
COPY <<'NGINX' /etc/nginx/conf.d/default.conf
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    # API -> backend container. Buffering must be OFF so answers stream.
    location /api/ {
        proxy_pass http://backend:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;
        proxy_read_timeout 180s;
    }

    location / {
        try_files $uri /index.html;
    }
}
NGINX
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
