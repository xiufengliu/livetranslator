<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/184Ti9p7uMRZP3K2jtPCr8xBZy7kwy2hv

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

## Deploy (Production)

This app is deployed as a static site served by Nginx. There is no Node process to run in production.

- Domain: `https://live.scicloud.site`
- Build output: `dist/` (served by Nginx)
- Deploy script: `./deploy-livetranslator`

Steps to deploy updates on the server:

1. SSH to the server and go to the app directory:
   `cd /opt/app/livetranslator`
2. Ensure your build-time secrets exist in `.env.local` (see below).
3. Run the deploy script:
   `./deploy-livetranslator`
4. If Nginx config changed, reload Nginx:
   `sudo nginx -t && sudo systemctl reload nginx`

Notes:
- The site is configured with security headers (HSTS, CSP, etc.) and long-term caching for hashed assets.
- The app uses the Screen Wake Lock API to keep the screen awake during recording (best-effort on mobile browsers).

## Secrets

Set your Gemini API key in `.env.local` (not committed):

```
GEMINI_API_KEY=YOUR_KEY_HERE
```

Rebuild after changing the key so it is embedded at build time:

```
npm run build
```
