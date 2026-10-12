# Harjot Singh Portfolio — Personal Server Edition

Complete standalone version of your animated portfolio and AI chatbot. It includes your résumé and preview, contact page, clickable project pages, downloadable project ZIPs, and Groq-backed conversational answers grounded in your supplied information. No ChatGPT Sites account is required to host it. No API key is included.

## Quick start on Windows, Linux, or macOS

1. Install Node.js 22 or newer from https://nodejs.org and extract this ZIP.
2. Open a terminal inside `Harjot_Portfolio_Server`.
3. Copy `.env.example` to `.env`:

   Windows PowerShell:
   ```powershell
   Copy-Item .env.example .env
   ```

   Linux/macOS:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` in a text editor and set:
   ```dotenv
   GROQ_API_KEY=your_new_groq_key_here
   GROQ_MODEL=openai/gpt-oss-20b
   ```
   Create a replacement key at https://console.groq.com/keys. The key previously posted in chat should be rotated. Do not put the key in frontend files or commit `.env`.
5. Run:
   ```bash
   npm start
   ```
6. Open **http://localhost:3000**. AI will activate automatically when a key is configured. No `npm install` or build step is needed: the server uses Node's built-in modules.

To open it from another device on your local network, visit `http://YOUR_SERVER_IP:3000`. The default HOST is `0.0.0.0`; permit port 3000 in your local firewall if needed. Browser voice input typically requires HTTPS or localhost; typed chat works without microphone access.

## Run with Docker

Install Docker with Compose, create `.env` as above, then run:

```bash
docker compose up -d --build
docker compose logs -f
```

The included Compose file binds to `127.0.0.1:3000`, suitable for a reverse proxy on the same server. Open http://localhost:3000 there. To allow LAN access, change the port mapping to `3000:3000`.

Stop with `docker compose down`. After editing `.env`, apply it with `docker compose up -d --force-recreate`.

## Put it on your own domain

1. Point your domain's DNS A record at your server's public IP.
2. Run the app with Node or Docker.
3. In `.env`, set `PUBLIC_ORIGIN=https://YOUR_DOMAIN` with no trailing slash, then restart the app.
4. Install Caddy on that server. Copy `Caddyfile.example` to your Caddy configuration, replace `portfolio.example.com` with your domain, and reload Caddy. Allow incoming ports 80 and 443 for HTTPS.

Caddy forwards to `127.0.0.1:3000` and supplies HTTPS. If you use another reverse proxy, forward to the same port and set `PUBLIC_ORIGIN` to the exact browser origin; otherwise same-origin checks may reject chat requests. Run `curl http://127.0.0.1:3000/health` to check server health. Your server needs outbound HTTPS access to `api.groq.com`.

For a Linux service without Docker, `portfolio.service.example` is provided. Create the `portfolio` service user, place the files at `/opt/harjot-portfolio`, confirm the Node binary path, and install the unit under `/etc/systemd/system/portfolio.service`. Keep `.env` readable only by the service user. Then run `sudo systemctl daemon-reload` and `sudo systemctl enable --now portfolio`.

## Optional password protection

Unlike the ChatGPT-hosted version, your personal server does not automatically restrict visitors. To keep the portfolio private, set BOTH in `.env` and use HTTPS:

```dotenv
BASIC_AUTH_USER=your_username
BASIC_AUTH_PASSWORD=your_strong_password
```

Leave both empty for a public portfolio. A configured server API key pays for/uses quota on visitor questions. Requests are limited to 15 per minute per direct client IP, in memory. With a reverse proxy, this conservative limiter groups visitors under the proxy IP. For a high-traffic public site, put an appropriate rate limit at your trusted reverse proxy. Limits reset on server restart and are not shared between server processes.

## What's included

- `public/` — animated HTML/CSS/JS frontend, résumé PDF and image preview, project downloads.
- `server.mjs` — standalone HTTP server, static asset serving, request limits, and optional authentication.
- `chat.mjs` — grounded AI generation, recent conversation, provider error handling, and model fallback.
- `data/knowledge.json` — personal and project facts used by the assistant.
- `projects/voice/` — Honey voice receptionist source, requirements, and Word knowledge base.
- `projects/churn/` — original churn dashboard, notebook, dataset, and requirements.
- `projects/spam/` — original SMS classifier, notebook, dataset, and requirements.
- Docker, Caddy, and systemd configuration examples.

The portfolio server does not execute the Streamlit or desktop voice apps. Their source downloads are available to visitors. To run those separately, use their own requirements and run `streamlit run app.py` in the appropriate project folder. The Honey script’s first-indexing return bug is corrected in this package; its desktop microphone/Whisper flow is independent of this website.

## Change your content

- Personal details and page content: `public/index.html`.
- Theme and animations: `public/theme.css`, `public/motion.js`.
- Chatbot facts: `data/knowledge.json` (restart the server after edits).
- System instructions/model behavior: `chat.mjs`.
- Updating your PDF: replace `public/harjot_singh_resume.pdf` and regenerate/replace `public/resume-preview.jpg`. Updating the PDF alone does not update the image preview or chatbot facts.

The chatbot sends the reference knowledge and recent conversation to Groq for AI generation. It does not use a local LLM or vector database. Grounding reduces unsupported claims but does not guarantee every generated answer is correct. Chat and session keys are held in browser memory rather than persisted.

## Checks and troubleshooting

```bash
npm test
```

Tests cover real local HTTP serving, project downloads, résumé preview, secret-file protection, optional authentication, and mocked AI integration. The deployed chatbot's live Groq responses were tested separately; your own key, provider quota, firewall, and HTTPS configuration must work on your server. Docker itself was not available for a runtime test in the packaging environment.

- Chat says key missing: check `.env`, spelling `GROQ_API_KEY`, and restart.
- Model unavailable: set `GROQ_MODEL=openai/gpt-oss-20b`; account availability can change.
- Quota exceeded: wait for Groq's limit to reset or inspect your account limits.
- Chat blocked after domain setup: check exact `PUBLIC_ORIGIN` and restart.
- Port busy: set another `PORT` and adjust your proxy/Compose mapping consistently.
- Microphone disabled: use HTTPS and grant browser microphone permission.
- Email app does not open: use Compose in Gmail or Copy email in Contact.
