# Sawti (OmniVoice web page + API) for a Hugging Face Docker Space with a GPU.
# Installs Sawti from GitHub at SAWTI_REF; the deploy workflow pins it to the
# commit being deployed.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# PyTorch wheels bundle CUDA 12.8; the Space provides the GPU driver.
RUN pip install --no-cache-dir --user \
    torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

ARG SAWTI_REPO=alfadni/Sawti_zato
ARG SAWTI_REF=master
RUN pip install --no-cache-dir --user \
    "omnivoice @ git+https://github.com/${SAWTI_REPO}@${SAWTI_REF}"

# Bake the weights into the image so a Space waking from sleep does not
# re-download several GB. Mirrors the lookups in OmniVoice.from_pretrained()
# and load_asr_model().
RUN python -c "import os; from huggingface_hub import snapshot_download as get; \
p = get('k2-fsa/OmniVoice'); \
os.path.isdir(os.path.join(p, 'audio_tokenizer')) or get('eustlb/higgs-audio-v2-tokenizer'); \
get('openai/whisper-large-v3-turbo')"

EXPOSE 7860
# SAWTI_API_KEY (a Space secret) is read from the environment by omnivoice-web;
# SAWTI_REQUIRE_API_KEY makes it refuse to start without one (fail closed).
ENV SAWTI_REQUIRE_API_KEY=1
CMD ["omnivoice-web", "--host", "0.0.0.0", "--port", "7860"]
