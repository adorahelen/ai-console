# =====================================================================
# ai-console app container — Ubuntu 22.04 + CUDA + Python 3.12 + venv
#
# 빌드 인자 (default 12.8 / cu128, --build-arg 로 13.0/cu130 도 지원):
#   --build-arg CUDA_VER=12.8.1   --build-arg TORCH_CUDA=cu128   (default)
#   --build-arg CUDA_VER=13.0.0   --build-arg TORCH_CUDA=cu130   (host driver 575+ 필요)
#
# 빌드 (CUDA 12.8):  docker build -t ai-console:cu128 -f docker/Dockerfile .
# 빌드 (CUDA 13.0):  docker build -t ai-console:cu130 -f docker/Dockerfile \
#                       --build-arg CUDA_VER=13.0.0 --build-arg TORCH_CUDA=cu130 .
# =====================================================================

ARG CUDA_VER=12.8.1
ARG TORCH_CUDA=cu128

FROM nvidia/cuda:${CUDA_VER}-cudnn-runtime-ubuntu22.04
# FROM 다음에는 ARG 가 stage-local 로 reset 되므로 다시 선언해야 RUN 에서 사용 가능
ARG TORCH_CUDA

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------
# (1) 시스템 패키지: Python 3.12 + 빌드 도구
#     deadsnakes PPA 가 ubuntu 22.04 에 python3.12 제공.
#     build-essential / cmake 는 sdist wheel (annoy, hnswlib 등) 빌드용.
# ---------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl wget git \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential cmake \
        libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------
# (2) Python venv + pip / setuptools / wheel
# ---------------------------------------------------------------------
RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------
# (3) torch — 큰 layer 라 별도 RUN 으로 두면 캐싱 효과. TORCH_CUDA build-arg 로 결정.
# ---------------------------------------------------------------------
RUN pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" torch==2.9.1

# ---------------------------------------------------------------------
# (4) requirements.deploy.txt — torch / llama_cpp_python / git+ 라인 제외
# ---------------------------------------------------------------------
WORKDIR /app
COPY requirements.deploy.txt /tmp/req.txt
RUN grep -v -E '^(torch==|llama_cpp_python|-e git\+)' /tmp/req.txt > /tmp/req.filtered.txt \
    && pip install -r /tmp/req.filtered.txt

# (4-1) spaCy 영어 모델 (presidio PII 게이트가 NLP 엔진으로 사용). ~560MB.
RUN python -m spacy download en_core_web_lg

# ---------------------------------------------------------------------
# (5) query-validator (로컬 editable)
# ---------------------------------------------------------------------
COPY query-validator /app/query-validator
RUN pip install -e /app/query-validator

# ---------------------------------------------------------------------
# (6) tiktoken_cache (오프라인 인코딩)
# ---------------------------------------------------------------------
COPY tiktoken_cache /app/tiktoken_cache

# ---------------------------------------------------------------------
# (7) 앱 코드 + prompts / docs (가장 자주 바뀌는 layer 마지막)
# ---------------------------------------------------------------------
COPY prompts /app/prompts
COPY docs /app/docs
COPY utils /app/utils
COPY *.py /app/
COPY aibotctl /app/aibotctl
COPY ai-agent.service /app/
RUN chmod +x /app/aibotctl /app/aibot_cli.py

# 모델 / config 는 volume 마운트로 주입
# - /models  → docker-compose 의 host 모델 디렉토리 (read-only)
# - /app/config.ini → host 의 config.ini

EXPOSE 8443

# 기본 실행 (run.sh 와 동일하게 qa_llm.py)
CMD ["python", "-u", "/app/qa_llm.py"]
