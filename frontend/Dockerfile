FROM node:22-alpine
WORKDIR /app
COPY package.json tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts index.html ./
COPY src src
RUN npm install
CMD ["npm", "run", "dev"]
