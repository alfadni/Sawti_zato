# Deploy Sawti to a Hugging Face Space

Runs the Sawti web page + API (`omnivoice-web`) on a GPU-backed
[Hugging Face Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker),
giving you a public HTTPS link such as `https://<user>-sawti.hf.space`.

Files involved:

| File | Purpose |
|---|---|
| `deploy/huggingface/Dockerfile` | Image: PyTorch (CUDA 12.8) + Sawti from GitHub + pre-downloaded model weights |
| `deploy/huggingface/README.md` | Space metadata (Docker SDK, port 7860, suggested T4 GPU) |
| `.github/workflows/deploy-hf-space.yml` | Creates/configures the Space (SDK, key, optional hardware and sleep) and uploads the two files above on every relevant push to `master` |

## One-time setup (≈ 10 minutes, no Hugging Face forms)

The workflow **creates the Space for you** (Docker SDK, public, gated by the
API key), syncs the key, and deploys. You only configure GitHub.

1. **Create a Hugging Face token** — <https://huggingface.co/settings/tokens> → *Create new token*
   - Simplest: type **Write**. (Fine-grained also works if it may create and write repos in your namespace.)

2. **Add GitHub secrets** — GitHub repo → *Settings* → *Secrets and variables* → *Actions* → *Secrets*
   - `HF_TOKEN` = the token from step 1
   - `SAWTI_API_KEY` = a long random string (e.g. `openssl rand -hex 24`) — users enter it in the page.
     Without it the container **refuses to start** (`SAWTI_REQUIRE_API_KEY=1`).

3. **Optional GitHub variables** — same page → *Variables* tab

   | Variable | Default | Example |
   |---|---|---|
   | `HF_SPACE` | `alfadni/sawti` | `your-username/sawti` |
   | `HF_SPACE_HARDWARE` | free CPU (too slow for real use) | `t4-small` — requires a payment method on Hugging Face |
   | `HF_SPACE_SLEEP` | Hugging Face default | `3600` (sleep after 1 h idle to cap GPU cost) |

   You can instead pick the hardware and sleep time in the Space's *Settings* page after the first deploy.

4. **Deploy** — GitHub repo → *Actions* → *Deploy to Hugging Face Space* → *Run workflow*.
   The Space builds the image (first build ≈ 10–20 min: PyTorch + ~5 GB of weights) and starts.

5. **Open** `https://huggingface.co/spaces/alfadni/sawti`, enter the API key in the page, and generate.

After this, every push to `master` that touches `omnivoice/`, `pyproject.toml`
or `deploy/huggingface/` redeploys automatically, pinned to that exact commit.

> **No "Docker" option on huggingface.co/new-space?** You don't need it. The
> Space type comes from `sdk: docker` in the uploaded `README.md`; if you
> created the Space by hand with another SDK, the first deploy switches it.

## Manual alternative (no GitHub Action)

Create a Space with any SDK, then upload `deploy/huggingface/Dockerfile` and
`deploy/huggingface/README.md` via *Files* → *Add file* → *Upload files* (the
README switches it to Docker), and add the `SAWTI_API_KEY` secret in the
Space settings. The Dockerfile then builds from the `master` branch.

## Operating notes

| Situation | What to expect / do |
|---|---|
| Space waking from sleep | 1–2 min while the model loads onto the GPU; weights are already in the image |
| Build fails at `pip install torch` | Check the PyTorch CUDA index is reachable from the build; retry the build |
| Space shows *Runtime error* right after start | Usually `SAWTI_API_KEY` is missing — logs say "refusing to start"; add the `SAWTI_API_KEY` GitHub secret and re-run the workflow |
| `401` in the page | Wrong key; the page asks for it again |
| `503 busy` | One request runs at a time per GPU; retry, or upgrade hardware / add replicas |
| CUDA out of memory | Use a larger GPU (e.g. L4 / A10G) |

## Security checklist before going public

- [ ] `SAWTI_API_KEY` set, long and random, shared only with intended users; rotate by updating the GitHub secret and re-running the workflow
- [ ] Space visibility reviewed (Private, or Public behind the API key)
- [ ] Sleep time set to cap cost
- [ ] Voice cloning used only with documented speaker consent
- [ ] No customer PII in test prompts (the server does not log text, but the browser keeps the last 5 results in the tab)
