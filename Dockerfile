FROM node:22-alpine
WORKDIR /app
COPY --chown=node:node package.json server.mjs chat.mjs ./
COPY --chown=node:node data ./data
COPY --chown=node:node public ./public
USER node
EXPOSE 3000
CMD ["node", "server.mjs"]
