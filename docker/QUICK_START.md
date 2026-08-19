# ai-console Quick Start

> 자세한 설치 / 트러블슈팅: [INSTALL_GUIDE.md](INSTALL_GUIDE.md)

## 사전 요구

- GPU: NVIDIA Ampere 이상 (L40S / A6000 / A100 / RTX 5090 등) — VRAM 24 GB+
- OS: Ubuntu 20.04 / 22.04 또는 RHEL 9 (Rocky/Alma 9 포함)
- RAM: 48 GB+ (4K docs 기준), 디스크 80 GB+
- 인터넷 (air-gap 은 INSTALL_GUIDE §1 참조)

---

## 1. 패키지 + 모델 옮기기 (빌드머신 → 타겟)

```bash
# 패키지 (~29 GB)
rsync -av ai-console-compose-<버전>/ user@host:/service/ai-console/

# 모델 (~22 GB) — bge-m3, gpt-oss-20b, llama-3.1-8b
rsync -av /path/to/models/ user@host:/service/models/
```

---

## 2. 타겟 호스트 — host SW 자동 설치

```bash
ssh user@host
cd /service/ai-console

sudo bash install_host_prereqs.sh -y       # driver / docker / toolkit 자동 설치
sudo reboot                                # driver 적용
```

reboot 후:

```bash
cd /service/ai-console
sudo bash install_host_prereqs.sh --skip-driver -y
```

마지막 줄에 **추천 CUDA tag (cu128/cu130)** 출력됨 → step 3 에 사용.

---

## 3. 환경 설정 — `.env` 만 편집

```bash
cp .env.example .env
vi .env
```

필수 편집:

```bash
MARIADB_ROOT_PASSWORD=<강한 비번>
MARIADB_DB_PASSWORD=<강한 비번>
AI_CONSOLE_TAG=cu128                    # step 2 에서 추천한 값
CUDA_VER=12.8.1                   # cu128=12.8.1 / cu130=13.0.0
TORCH_CUDA=cu128
SERVER_PORT=8443                  # 외부 노출 포트
```

> `config.ini.docker` 는 install_compose.sh 가 자동으로 생성 (default 그대로 동작 OK).
> 컨텍스트 크기 / batch / parallel 등 튜닝 시에만 미리 `cp config.ini.docker.example config.ini.docker && vi config.ini.docker`.

---

## 4. 앱 기동

```bash
sudo bash install_compose.sh
```

5 컨테이너 (mariadb, qdrant, llama-main, llama-translation, app) 기동 + BGE 인덱싱.
성공 시 마지막에 **API key + URL** 출력.

---

## 5. 동작 확인

```bash
# API key 확인
./manage_keys.sh show-default

KEY=<위 출력값>
HOST=$(hostname -I | awk '{print $1}')
PORT=$(grep ^SERVER_PORT .env | cut -d= -f2)

# 헬스 체크
curl -k -H "Authorization: Bearer $KEY" https://$HOST:$PORT/api/ai/hello
# → {"is_valid":true}

# 실 질의
curl -k -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
     -d '{"question":"안녕"}' \
     https://$HOST:$PORT/api/ai/chats
```

---

## 운영 명령 (`manage_stack.sh`)

```bash
./manage_stack.sh status                   # 컨테이너 상태
./manage_stack.sh logs app                 # 앱 로그 follow
./manage_stack.sh restart app              # 앱만 재시작
./manage_stack.sh health                   # /api/ai/hello
./manage_stack.sh gpu                      # nvidia-smi 요약
./manage_stack.sh down                     # 전체 중지 (데이터 유지)
```

## API 키 관리 (`manage_keys.sh`)

```bash
./manage_keys.sh show-default              # 기본 키
./manage_keys.sh generate <name> <account> # 신규 발급
./manage_keys.sh list                      # 키 목록 (마스킹)
./manage_keys.sh renew <api_key>           # 만료 +1년
./manage_keys.sh delete <api_key>          # 삭제
```

---

## 막혔을 때

```bash
# 받아야 할 .deb/.rpm 목록 (air-gap)
bash install_host_prereqs.sh --info

# 컨테이너 시작 안 될 때
docker compose logs app --tail 100
docker compose logs llama-main --tail 50

# GPU 점유 확인
nvidia-smi

# 재기동
docker compose down && docker compose up -d
```

추가 트러블슈팅 → [INSTALL_GUIDE.md §7](INSTALL_GUIDE.md)
