# ai-console Docker Deploy — 설치 가이드

`build_images.sh` 가 만든 패키지를 타겟 서버에 옮겨 air-gap 환경에서도 기동할 수 있는 절차.

---

## 0. 하드웨어 요구사항

### GPU
| 항목 | 최소 | 권장 |
|---|---|---|
| GPU VRAM | 24 GB | **48 GB** (L40S, A6000, A100-40, RTX 6000 Ada) |
| CUDA capability | sm_80 (Ampere) | sm_86 / sm_89 / sm_90 / sm_120 |

VRAM 사용 내역:
- gpt-oss-20b (메인 LLM, F16 GGUF): ~14 GB
- llama-3.1-8b-q4_k_m (번역용): ~7 GB
- BGE-M3 임베딩 모델 (fp16): ~2.3 GB
- 합계 idle: ~24 GB / 인덱싱 시 일시적으로 +1~2 GB

**24 GB GPU (RTX 4090 등)**: 동작은 하나 여유 부족. 번역 LLM 안 쓰면 가능.
**40 GB+ GPU (A100-40, L40S 46GB)**: 권장. 배치 크기 / context 늘릴 여유.

### CPU + RAM ⚠️ **중요**
| docs 수 | 최소 RAM | 권장 RAM |
|---|---|---|
| ~2,000 | 32 GB | 48 GB |
| **~4,000 (기본)** | **48 GB** | **64 GB** |
| ~10,000 | 96 GB | 128 GB |
| ~20,000 | 160 GB | 192 GB |

**왜 RAM 이 이렇게 큰가**: BGE-M3 인덱싱 시 dense + sparse + colbert (192 토큰 × 1024 차원) 임베딩을 Python list 로 메모리에 누적한 뒤 한 번에 Qdrant 에 upsert. doc 당 약 4.8 MB Python 객체 메모리 점유. 데이터셋과 RAM 이 거의 선형 관계.

> **swap 대체 불가**: swap 으로 buff/cache 압박을 받으면 인덱싱이 매우 느려짐 (~3 분 → 15 분+). 가능하면 물리 RAM 으로 확보.

| 항목 | 최소 | 권장 |
|---|---|---|
| CPU | 4 vCPU | **8 vCPU** |
| 디스크 (서버 전체) | 80 GB | **120 GB** SSD |
| 네트워크 | — | 최초 모델/이미지 옮길 때만 사용 |

### AWS / GCP / Azure 인스턴스 가이드
| 클라우드 | 인스턴스 | GPU | vCPU | RAM | 적합 |
|---|---|---|---|---|---|
| AWS | g6e.xlarge | L40S 46GB | 4 | 32 GB | ❌ ~2K docs |
| AWS | **g6e.2xlarge** | **L40S 46GB** | **8** | **64 GB** | ✅ **~6K docs (권장)** |
| AWS | g6e.4xlarge | L40S 46GB | 16 | 128 GB | ✅✅ ~13K docs |
| AWS | p5.48xlarge | H100 80GB×8 | 192 | 2 TB | overkill |
| GCP | g2-standard-16 | L4 24GB | 16 | 64 GB | ⚠️ VRAM 빠듯 |
| GCP | g2-standard-32 | L4 24GB | 32 | 128 GB | ⚠️ VRAM 빠듯 |

> 데이터셋 늘릴 계획이면 RAM 을 그에 맞춰 산정. 코드 측 BGE colbert 누적 패턴이 변경되면 위 수치는 1/4~1/8 수준으로 감소 가능 (별도 개선 과제).

---

## 0a. 패키지 구성

```
ai-console-compose-<version>/
├── docker-compose.yml              # 4 services 정의
├── install_host_prereqs.sh         # host SW(driver/docker/toolkit) 자동 설치 (Ubuntu/RHEL9)
├── install_compose.sh              # 타겟 서버 설치 스크립트
├── .env.example                    # 환경변수 템플릿
├── config.ini.docker.example       # 앱 설정 템플릿 (plain text)
├── INSTALL_GUIDE.md                # 본 문서
├── README.md                       # 짧은 요약
├── images/
│   ├── mariadb.tar                 #  ~330 MB
│   ├── qdrant.tar                  #  ~190 MB
│   ├── ai-console-cu128.tar              # ~13 GB
│   ├── ai-console-cu130.tar              #  ~9 GB
│   ├── ai-console-llama-server-cu128.tar # ~4.6 GB
│   └── ai-console-llama-server-cu130.tar # ~3 GB
├── ssl/                            # self-signed TLS 인증서
├── api_keys/                       # 빈 디렉토리 (런타임에 채워짐)
├── cache/                          # 빈
├── logs/                           # 빈
└── utils-docker/                   # 빈 (.encryption_key 들어갈 자리)
```

**모델은 패키지에 포함 안 됨** (~22 GB). 별도 USB / scp 로 옮겨야 함.

---

## 1. 사전 요구 — 타겟 host SW

위 §0 의 하드웨어 요구사항이 충족된 상태에서, host OS 에 아래 SW 를 미리 설치해야 함.

`install_compose.sh` 는 검증만 하고 실패하면 멈춤. 아래는 **반드시 host 에 미리** 깔려있어야 함.

> 💡 **자동 설치 스크립트**: Ubuntu 20.04 / 22.04 / RHEL 9 (Rocky/Alma 포함) 라면 `install_host_prereqs.sh` 가 §1.1~1.4 를 한 번에 처리.
> ```bash
> sudo bash install_host_prereqs.sh -y                  # GPU 자동 감지 (cu128 기준)
> sudo bash install_host_prereqs.sh --cuda cu130 -y     # cu130 이미지 쓸 거면 (driver 575 강제)
> sudo bash install_host_prereqs.sh --driver-version 535 -y  # 자동 감지 무시 강제
> # → 드라이버 신규 설치 시 reboot 안내, 재부팅 후:
> sudo bash install_host_prereqs.sh --skip-driver -y
> ```
> driver 자동 선택 정책: `--cuda cu130` 이면 GPU 무관 575 / cu128 면 Blackwell→575, Ampere/Ada/Hopper→535.
>
> **오프라인 / air-gap**: 인터넷 미감지 시 자동으로 다운로드 가이드 출력 후 종료. 강제 출력은 `--info`:
> ```bash
> bash install_host_prereqs.sh --info             # GPU/OS 감지 → 추천 CUDA + 받아야 할 .deb/.rpm 목록 출력
> ```

### 1.1 NVIDIA driver

| 이미지 tag | 최소 driver | 권장 |
|---|---|---|
| `cu128` | 535+ | 550 / 555 / 570 |
| `cu130` | 575+ | 575+ |

```bash
# Ubuntu
sudo apt install nvidia-driver-575 && sudo reboot

# RHEL / Rocky / Alma
sudo dnf install nvidia-driver && sudo reboot

# .run 인스톨러 (배포판 무관)
sudo bash NVIDIA-Linux-x86_64-575.xx.run --silent
```

확인:
```bash
nvidia-smi
# Driver Version: 575.xx  CUDA Version: 13.x  ← 표시되면 OK
```

### 1.2 Docker 24+

```bash
# Ubuntu (공식 docker.io)
curl -fsSL https://get.docker.com | sudo bash

# RHEL
sudo dnf install docker-ce docker-ce-cli containerd.io
sudo systemctl enable --now docker

# 사용자를 docker 그룹에 추가 (sudo 없이 docker 쓰려면)
sudo usermod -aG docker $USER
newgrp docker
```

### 1.3 nvidia-container-toolkit

```bash
# Ubuntu
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 검증
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi
```

### 1.4 docker compose plugin v2+

```bash
# Ubuntu
sudo apt install docker-compose-plugin

# 또는 user-local (sudo 없이)
mkdir -p ~/.docker/cli-plugins
wget -O ~/.docker/cli-plugins/docker-compose \
    https://github.com/docker/compose/releases/download/v2.32.1/docker-compose-linux-x86_64
chmod +x ~/.docker/cli-plugins/docker-compose

# 검증
docker compose version
```

> **air-gap 환경**이면 위 인터넷 의존 명령들이 안 됨. driver `.run` / docker-ce `.deb` 또는 `.rpm` / nvidia-container-toolkit `.deb` 등을 빌드머신에서 미리 받아 USB 에 같이 넣어야 함.

---

## 2. 모델 디렉토리 준비

타겟 host 에 모델 트리 배치:

```
/service/models/                    # 기본 (.env 의 MODELS_DIR)
├── bge-m3/                         # BGE-M3 임베딩 모델 (~2.3 GB)
├── gpt-oss-20b-GGUF/
│   └── gpt-oss-20b-F16.gguf        # 메인 LLM (~13 GB)
└── meta-llama-3.1-8b-instruct-q4_k_m.gguf   # 번역용 (~4.7 GB)
```

```bash
# USB 에서 옮기는 예
sudo mkdir -p /service/models
sudo cp -r /mnt/usb/models/* /service/models/

# read-only 마운트 되니 권한은 그대로 둬도 됨
ls -la /service/models/
```

---

## 3. 패키지 배치

```bash
# USB / scp 로 옮긴 패키지 풀어 두기
cp -r ai-console-compose-20260506-XXXXXXX/ /opt/ai-console
cd /opt/ai-console
```

또는 tar.gz 로 받았다면:
```bash
tar xzf ai-console-compose-20260506-XXXXXXX.tar.gz -C /opt/
mv /opt/ai-console-compose-20260506-XXXXXXX /opt/ai-console
cd /opt/ai-console
```

---

## 4. 환경 / 설정 파일 편집

### 4.1 `.env`

```bash
cp .env.example .env
vi .env
```

필수 편집 항목:

```bash
# DB 비번 (production 에선 바꿔야)
MARIADB_ROOT_PASSWORD=<강한 비번>
MARIADB_DB_PASSWORD=<강한 비번>

# 모델 위치
MODELS_DIR=/service/models

# 외부 노출 포트 (FastAPI HTTPS)
SERVER_PORT=8443

# CUDA tag — 셋이 일치해야 함
AI_CONSOLE_TAG=cu128         # 또는 cu130
CUDA_VER=12.8.1        # cu128 → 12.8.1, cu130 → 13.0.0
TORCH_CUDA=cu128       # AI_CONSOLE_TAG 와 동일

# OpenAI 사용 시
OPENAI_API_KEY=sk-...

# Slack 사용 시
SLACK_ENABLED=true
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...
```

### 4.2 `config.ini.docker`

```bash
cp config.ini.docker.example config.ini.docker
# 평문 그대로. 첫 docker compose up 시 ENC: 자동 변환됨.
```

대부분 default 값 그대로 사용 가능. 바꿀 일이 있다면:
- `[server] port` (container 내부 포트, default 5443 — `.env` 의 `SERVER_PORT` 와 매핑)
- `[llama_server] n_ctx`, `n_parallel` (GPU VRAM 부족 시 줄임)

---

## 5. 설치 실행

```bash
sudo bash install_compose.sh
```

스크립트가 6단계로 진행:

| 단계 | 동작 | 예상 시간 |
|---|---|---|
| 1 | 사전 요구 검증 (driver / docker / toolkit / compose) | 즉시 |
| 2 | 이미지 4~6개 `docker load` | 1~3분 |
| 3 | `.env` / `config.ini.docker` 검증 | 즉시 |
| 4 | 모델 디렉토리 검증 | 즉시 |
| 5 | `docker compose up -d` (5 컨테이너) + uvicorn ready 대기 | 30초~2분 |
| 6 | `init_system.py` (BGE 인덱싱) | 3~5분 |

옵션:

```bash
# 이미지 이미 load 됨 (재실행 시)
sudo bash install_compose.sh --skip-load

# init 안 돌림 (기존 인덱싱 재사용)
sudo bash install_compose.sh --skip-init

# MODELS_DIR 다른 경로
sudo bash install_compose.sh --models-dir /data/models
```

성공 시 마지막에 API key + 접속 URL 출력:

```
============================================================
   ✅ 설치 완료
============================================================
   서비스 상태  : docker compose ps
   로그         : docker compose logs -f app
   외부 접속    : https://<host>:8443/docs
   API key      : 12345678-90ab-cdef-1234-567890abcdef

   curl 예시:
     curl -k -H "Authorization: Bearer ..." \
          https://localhost:8443/api/ai/hello
============================================================
```

---

## 6. 운영

> 같은 디렉토리의 **`manage_stack.sh`** (스택 운영) / **`manage_keys.sh`** (API 키) 가 아래 모든 명령을 단축해줌. 자세한 raw `docker compose` 명령은 그대로 사용해도 무방.

### 6.0 manage_stack.sh — 운영 명령 단축

```bash
cd /opt/ai-console
./manage_stack.sh --help     # 전체 명령

# 자주 쓰는 것들
./manage_stack.sh status                   # docker compose ps
./manage_stack.sh logs app                 # 로그 follow
./manage_stack.sh restart app              # 재시작
./manage_stack.sh shell app                # 컨테이너 bash
./manage_stack.sh db                       # mariadb 접속
./manage_stack.sh health                   # /api/ai/hello 확인
./manage_stack.sh gpu                      # nvidia-smi 요약
./manage_stack.sh init                     # BGE 인덱싱 재실행
./manage_stack.sh down                     # 컨테이너 제거 (DB 유지)
./manage_stack.sh purge                    # ⚠️ 볼륨까지 (DB 삭제, 확인 프롬프트)
```

### 6.1 상태 / 로그 (raw docker compose)

```bash
cd /opt/ai-console

docker compose ps
docker compose logs -f app
docker compose logs -f llama-main
docker compose logs -f llama-translation
docker compose logs -f mariadb
docker compose logs -f qdrant

# 특정 컨테이너 shell
docker exec -it ai-console-app bash
docker exec -it ai-console-mariadb mariadb -uagent -p<MARIADB_DB_PASSWORD> agent
```

### 6.1.1 API 키 관리 (`manage_keys.sh`)

기존 `run.sh` 의 generate-key/list-keys/verify-key 등을 docker 환경용으로 재패키징한 스크립트. 같은 디렉토리의 `.env` 에서 SERVER_PORT 를 읽어 호스트의 `https://localhost:<SERVER_PORT>` 로 접속.

```bash
cd /opt/ai-console

# default 구독 키 (install_compose 가 만든 자동 키 — DB 직조회)
./manage_keys.sh show-default

# 새 키 발급
./manage_keys.sh generate <name> <account> [description] [acl]
./manage_keys.sh generate john dev_team "Dev user" "192.168.1.0/24"

# 키 목록 (마스킹)
./manage_keys.sh list
./manage_keys.sh list <name> [account]   # 검색

# 키 검증
./manage_keys.sh verify <api_key>

# 만료일 연장 (기본 +1년)
./manage_keys.sh renew <api_key>
./manage_keys.sh renew <api_key> 100             # +100년
./manage_keys.sh renew <api_key> 30 day          # +30일
./manage_keys.sh renew <api_key> 6 month         # +6개월
# duration 범위 1~100, unit: day | month | year (서버 측 제한)

# 키 삭제 (인터랙티브 확인)
./manage_keys.sh delete <api_key>
```

**환경변수**:
- `ADMIN_KEY`: 운영에서 admin 검증을 켰다면 필요 (기본: 32자 더미. 서버가 admin 검증 비활성 상태면 더미로도 통과)
- `AI_CONSOLE_HOST`: 서버 host (기본 `localhost`. 다른 머신에서 호출하면 도메인/IP 지정)

```bash
# 다른 머신에서 호출
AI_CONSOLE_HOST=10.0.0.5 ./manage_keys.sh list

# admin 검증 켠 운영 환경
ADMIN_KEY=$(cat ~/admin.key) ./manage_keys.sh delete xxxx-yyyy-...
```

---

### 6.2 재시작 / 중지

```bash
docker compose restart app           # 앱만
docker compose restart               # 전체
docker compose stop                  # 중지
docker compose up -d                 # 재기동
docker compose down                  # 컨테이너 제거 (볼륨은 유지)
docker compose down -v               # 볼륨까지 삭제 (DB / Qdrant 데이터 날아감 — 주의)
```

### 6.3 업데이트 (새 패키지 받았을 때)

```bash
cd /opt/ai-console
docker compose down

# 새 이미지 tar 만 교체
cp /mnt/usb/ai-console-compose-NEW/images/*.tar images/
for tar in images/*.tar; do docker load -i "$tar"; done

# 새 compose / install 스크립트도 교체 (.env / config 는 보존)
cp /mnt/usb/ai-console-compose-NEW/{docker-compose.yml,install_compose.sh,INSTALL_GUIDE.md} .

docker compose up -d
```

### 6.4 CUDA 버전 변경 (cu128 ↔ cu130)

`.env` 편집:
```
AI_CONSOLE_TAG=cu130
CUDA_VER=13.0.0
TORCH_CUDA=cu130
```

```bash
docker compose down
docker compose up -d
```

driver 가 575+ 인지 먼저 확인 (`nvidia-smi`).

---

## 7. 트러블슈팅

### 7.1 `install_compose.sh [1/6]` 단계에서 실패
- `NVIDIA driver 없음` → 1.1 절 참고
- `driver < 535` → driver 업그레이드 후 reboot
- `docker 미설치` → 1.2 절
- `docker compose plugin 미설치` → 1.4 절
- `nvidia-container-toolkit 미설치` → 1.3 절

### 7.2 `docker load` 가 너무 느림
- 디스크 IO 병목. SSD 권장. NFS 위에서는 매우 느림 → 로컬 디스크에 풀어두고 실행.

### 7.3 `docker compose up` 직후 `ai-console-app` 이 죽음
```bash
docker compose logs app | tail -100
```
자주 발생하는 원인:
- **MariaDB 미준비**: `docker compose ps` 로 mariadb healthy 확인. 안 되면 30초 더 기다린 후 `docker compose up -d`.
- **Qdrant 미준비**: `docker exec ai-console-app curl http://qdrant:6333/`
- **모델 경로 오류**: `MODELS_DIR` 안에 `bge-m3/`, `gpt-oss-20b-GGUF/` 가 있는지
- **GPU 점유**: 호스트에 다른 프로세스가 GPU 쓰면 OOM. `nvidia-smi` 로 확인.
- **`.encryption_key` 가 디렉토리** (`IsADirectoryError`): host 의 `utils-docker/.encryption_key` 가 없으면 docker 가 디렉토리로 자동 생성하는 footgun.
  → `install_compose.sh` 가 자동으로 32-byte 키 생성하지만, 수동 복구 시:
  ```bash
  cd /opt/ai-console
  docker compose stop app
  sudo rm -rf utils-docker/.encryption_key
  sudo python3 -c "import os; open('utils-docker/.encryption_key','wb').write(os.urandom(32))"
  sudo chmod 600 utils-docker/.encryption_key
  docker compose up -d app
  ```
- **`.encryption_key` 가 빈 파일** (`ZeroDivisionError: integer division by zero` in `_xor_encrypt_decrypt`): 위와 동일하게 32-byte 키 채워줌.

### 7.4 GPU OOM (`CUDA out of memory`)
- 메인 LLM (`llama-main`): `docker-compose.yml` 의 `command` 에서 `-c 65536` → `-c 32768`, `-np 8` → `-np 4`
- BGE 임베딩: `config.ini.docker` 의 `[embedding].batch_size` 낮춤
- `docker compose down && docker compose up -d` 로 적용

### 7.5 외부에서 접속 안 됨
```bash
# 호스트에서 직접 확인
curl -k https://localhost:8443/api/ai/hello

# 안 되면 컨테이너까지 내려가서
docker exec ai-console-app curl -k https://127.0.0.1:5443/api/ai/hello
```
- 방화벽 / SG 에서 SERVER_PORT (default 8443) 열려있는지
- 다른 프로세스가 SERVER_PORT 점유 중이면 `.env` 의 `SERVER_PORT` 변경

### 7.6 BGE 인덱싱 (`init_system.py`) 실패
```bash
docker exec ai-console-app python /app/utils/init_system.py
```
- Qdrant 응답 대기 시간 부족: `docker compose logs qdrant` 확인 후 재실행
- 임베딩 GPU 부족: 다른 GPU 컨테이너 잠시 stop

### 7.7 자동 시작 (서버 재부팅 후)
`docker-compose.yml` 의 모든 service 가 `restart: unless-stopped` 라 docker 가 살아있으면 자동 기동됨.

```bash
sudo systemctl enable docker     # docker 자체가 부팅 시 시작되도록
```

---

## 8. 완전 제거

```bash
cd /opt/ai-console
docker compose down -v             # 컨테이너 + 볼륨(DB / Qdrant 데이터) 제거

docker rmi ai-console:cu128 ai-console:cu130 ai-console-llama-server:cu128 ai-console-llama-server:cu130
docker rmi mariadb:11 qdrant/qdrant:v1.12.1

cd /
sudo rm -rf /opt/ai-console
sudo rm -rf /service/models        # 모델까지 지울 거면
```

---

## 9. CUDA 버전 선택 가이드

| 환경 | 권장 tag | 이유 |
|---|---|---|
| 기존 운영 GPU (A100 / L40S / RTX 4090 + driver 535~570) | `cu128` | 안정적, driver 호환 폭 넓음 |
| 신규 도입 GPU (RTX 5090 / Blackwell + driver 575+) | `cu130` | sm_120 native 지원, 이미지 30% 작음 |
| 혼재 환경 (서버마다 driver 다름) | 각각 따로 | `.env` 에서 AI_CONSOLE_TAG 만 달리하면 됨 |

빌드머신에서 두 버전 다 만든 경우 패키지에 양쪽 tar 포함됨. 타겟에서 `.env` 만 바꿔서 선택.

---

## 부록 A. 외부 access 포트 정리

| 포트 | 노출 | 용도 |
|---|---|---|
| `${SERVER_PORT}` (default 8443) | host → app:5443 | FastAPI HTTPS |
| 3306 | container 내부만 | MariaDB |
| 6333 / 6334 | container 내부만 | Qdrant REST / gRPC |
| 8181 | container 내부만 | llama-main / llama-translation |

외부에 노출되는 건 SERVER_PORT 하나뿐. 다른 포트 노출하려면 `docker-compose.yml` 의 service 에 `ports:` 추가.

## 부록 B. 디스크 사용량

| 항목 | 사이즈 |
|---|---|
| 패키지 압축전 | ~30 GB |
| docker load 후 (이미지 저장소) | ~30 GB |
| 모델 파일 | ~22 GB |
| Qdrant 인덱싱 데이터 | ~1 GB (3,800 docs 기준) |
| MariaDB 데이터 | ~50 MB |
| logs / cache | ~수백 MB / 일 |

**최소 디스크: 80 GB 권장**.
