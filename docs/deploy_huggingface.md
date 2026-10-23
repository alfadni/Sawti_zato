# Deploy Sawti to a Hugging Face Space

Runs the Sawti web page + API (`omnivoice-web`) on a GPU-backed
[Hugging Face Docker Space](https://huggingface.co/docs/hub/spaces-sdks-docker),
giving you a public HTTPS link such as `https://<user>-sawti.hf.space`.

Files involved:

| File | Purpose |
|---|---|
| `deploy/huggingface/Dockerfile` | Image: PyTorch (CUDA 12.8) + Sawti from GitHub + pre-downloaded model weights |
| `deploy/huggingface/README.md` | Space metadata (Docker SDK, port 7860, suggested T4 GPU) |
| `.github/workflows/deploy-hf-space.yml` | Pushes the two files above to the Space on every relevant push to `master` |

## One-time setup (≈ 15 minutes)

1. **Create the Space** — <https://huggingface.co/new-space>
   - Owner/name: e.g. `your-username/sawti`
   - SDK: **Docker** → *Blank*
   - Hardware: **a GPU** (e.g. *Nvidia T4 small*). CPU hardware will start but is far too slow for real use.
   - Visibility: *Private* while you test; switch to *Public* when ready.

2. **Set the API key on the Space** — Space → *Settings* → *Variables and secrets* → *New secret*
   - Name: `SAWTI_API_KEY`, value: a long random string (e.g. `openssl rand -hex 24`).
   - The container **refuses to start without it** (`SAWTI_REQUIRE_API_KEY=1`), so voice cloning is never exposed unauthenticated.

3. **Set the sleep time** — Space → *Settings* → *Sleep time*: e.g. 1 hour, so the GPU stops billing when idle.

4. **Create a Hugging Face token** — <https://huggingface.co/settings/tokens>
   - Type *Fine-grained*, with write access to the Space only.

5. **Connect GitHub** — GitHub repo → *Settings* → *Secrets and variables* → *Actions*
   - *Secrets* tab → `HF_TOKEN` = the token from step 4
   - *Variables* tab → `HF_SPACE` = `your-username/sawti`

6. **Deploy** — GitHub repo → *Actions* → *Deploy to Hugging Face Space* → *Run workflow*.
   The Space then builds the image (first build ≈ 10–20 min: PyTorch + ~5 GB of weights) and starts.

7. **Open** `https://huggingface.co/spaces/your-username/sawti`, enter the API key in the page, and generate.

After this, every push to `master` that touches `omnivoice/`, `pyproject.toml`
or `deploy/huggingface/` redeploys automatically, pinned to that exact commit.

## Manual alternative (no GitHub Action)

Upload `deploy/huggingface/Dockerfile` and `deploy/huggingface/README.md` to
the Space via *Files* → *Add file* → *Upload files*. The Dockerfile then builds
from the `master` branch.

## Operating notes

| Situation | What to expect / do |
|---|---|
| Space waking from sleep | 1–2 min while the model loads onto the GPU; weights are already in the image |
| Build fails at `pip install torch` | Check the PyTorch CUDA index is reachable from the build; retry the build |
| Space shows *Runtime error* right after start | Usually `SAWTI_API_KEY` is missing — logs say "refusing to start" |
| `401` in the page | Wrong key; the page asks for it again |
| `503 busy` | One request runs at a time per GPU; retry, or upgrade hardware / add replicas |
| CUDA out of memory | Use a larger GPU (e.g. L4 / A10G) |

## Security checklist before going public

- [ ] `SAWTI_API_KEY` set, long and random, shared only with intended users; rotate by updating the secret (the Space restarts)
- [ ] Space visibility reviewed (Private, or Public behind the API key)
- [ ] Sleep time set to cap cost
- [ ] Voice cloning used only with documented speaker consent
- [ ] No customer PII in test prompts (the server does not log text, but the browser keeps the last 5 results in the tab)
