# syntax=docker/dockerfile:1
FROM oven/bun:1.3-alpine AS base
WORKDIR /app
COPY package.json bun.lock* ./
RUN --mount=type=cache,target=/root/.bun/install/cache \
    bun install

FROM base AS build
COPY . .
ARG VITE_ABLY_KEY
ENV VITE_ABLY_KEY=$VITE_ABLY_KEY
RUN bun run build

FROM base AS dev
COPY . .
EXPOSE 3000
CMD ["bun", "run", "dev", "--", "--host", "0.0.0.0"]

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
