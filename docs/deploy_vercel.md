# Serve the Sawti page from Vercel (model stays on Hugging Face)

Vercel cannot run the model (no GPUs; serverless functions are far too small
for PyTorch + ~5 GB of weights). It can serve the page on a custom domain and
forward API calls to the GPU backend on the Hugging Face Space:

```
Browser ──► https://sawti.<your-domain> (Vercel: static page + security headers)
                    │ /api/*  (Vercel rewrite, same origin → no CORS, no key in config)
                    ▼
            https://alfadni-sawti.hf.space (Hugging Face Space, GPU, omnivoice-web)
```

**Prerequisite:** the Space from [deploy_huggingface.md](deploy_huggingface.md)
is running, set to **Public**, with `SAWTI_API_KEY` configured. A Private Space
cannot be reached by a Vercel rewrite (it would need a Hugging Face token on
every request).

## Setup (≈ 5 minutes)

1. If your Space is not `alfadni/sawti`, edit the `destination` in
   [`deploy/vercel/vercel.json`](../deploy/vercel/vercel.json):
   `https://<owner>-<space>.hf.space/api/:path*` (lowercase; `_` and `.` become `-`).
2. Vercel → **Add New… → Project** → import `alfadni/Sawti_zato`.
3. **Root Directory:** `deploy/vercel` (keep "Include files outside the Root
   Directory" enabled — the build copies `omnivoice/web/static/index.html`).
   Framework preset: **Other**. Leave build settings as they come from `vercel.json`.
4. **Deploy.** Optionally add your domain under *Settings → Domains*.

Every push to `master` then redeploys the page; pull requests get preview URLs.

## What `vercel.json` does

- Build: copies the single page into `public/` — no dependencies installed.
- Rewrite: `/api/*` → the Space, so the page works unchanged (same-origin calls).
- Headers: strict Content-Security-Policy (only same-origin requests, no
  framing), `nosniff`, `no-referrer`, HSTS, and camera/microphone/geolocation
  disabled.

## Behaviour and limits

| Situation | What users see |
|---|---|
| Space asleep / building | Status pill shows "Starting up…" and retries every 15 s; generation returns "starting up or busy" |
| Missing / wrong API key | Page asks for the key; `401` from the Space |
| Space busy | `503` → "try again in a minute" |
| Very long generations | Vercel's proxy has its own timeout for external rewrites; typical requests finish in seconds |

## Security notes

- The API key is never stored in Vercel or in the repo: users type it in the
  page (kept in the browser tab's session storage only).
- The Space URL stays reachable directly; the API key — not the Vercel domain —
  is the access control.
- For named-user login (SSO), audit trails or per-user rate limits, add an
  identity provider and a gateway in front of the Space; a rewrite alone
  cannot do that.
