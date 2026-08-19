# 개발 방향 — ai-console

## 정체성

범용 온프레미스 AI 에이전트 **콘솔(본체)**. 엔진은 고정하고, 도메인은 카트리지(프롬프트·지식·모델 3슬롯)로 꽂는다.

- **계보**: 실운영 SecOps 관제 에이전트의 프로덕션 검증된 아키텍처에서 출발해, 비공개 상위 리포에서 도메인 탈색을 마친 뒤 공개용으로 분리했다. **이 리포는 공개(public) 전제** — 내부 인프라 식별자·독점 도메인 자산을 커밋하지 않는다.
- **최종 목표**: 누구나 **셸 스크립트 한 번으로 설치**하고, web-UI에서 클릭-클릭으로 자기 도메인의 AI 에이전트를 구축하는 제품.

## 북극성 (바꾸지 않는 방향)

1. **아키텍처 유지** — FastAPI + llama-server(GGUF) + Qdrant/BGE-M3 2-way RRF + intent 분류 + 핸들러 게이트웨이. 새 프레임워크(LangChain류) 도입 금지.
2. **카트리지 = 기존 통로 재사용.** 새 메커니즘을 발명하지 않는다:
   - 프롬프트 → `config.ini [prompts]` (경로 전부 외부화)
   - 지식 → `POST /api/ai/prompts/bulk` (YAML→임베딩→Qdrant)
   - 모델 → `[model] model=` + `models.yaml` 프리셋
3. **모델은 HW 취사선택** — `models.yaml` 티어(cpu-only→8GB→16GB→24GB+→api)가 단일 소스. 프리셋 추가는 다운로드 명령+VRAM 실측치 필수.
   티어 기본은 **Gemma 4 계열로 통일**(Gemma-first) — 전 티어 `runtime=server`가 되어 llama-cpp-python 빌드가 빠지고 KV 캐시 재사용이 기본으로 붙는다. gpt-oss·llama는 `--preset` 전용으로 유지.
   후보 선별은 `min_vram_gb`·`min_ram_gb` **AND 조건**이며, 0건이면 설치를 강행하지 않고 중단한다. HW→모델 매핑은 README 표가 계약이고 CI가 11행을 고정한다.
4. **파인튜닝 없음** — 도메인 능력은 프롬프트+RAG로만 만든다. 순정 가중치 원칙.

## 현황·로드맵

- Phase 1 도메인 탈색 ✅ / Phase 2 install.sh ✅ / Phase 3 web 위저드 ✅
- **Phase 3.5 동적 검증 (진행 중)**: 절차·성공기준은 [docs/testing-guide.md](docs/testing-guide.md). T1(CPU-only VM)·T2(16GB GPU 데스크톱) 경로.
- **Phase 4 카트리지 생태계**: `cartridge.yaml` 스키마 검증, 장착/해제 CLI(`aibotctl` 확장), 공개 가능한 예시 카트리지 제작.
## 공개 전환 (2026-07-31 기준 잔여)

공개 전 체크리스트는 전부 종결됐고, 이 리포는 **새 히스토리로 시작**했다(초기 커밋 1개).
남은 것은 셋뿐이다.

1. **⛔ 설치 스모크 1회 (선행 조건)** — 깨끗한 VM에서 `./install.sh` 실설치 완주 + 위저드·RAG
   종단([docs/testing-guide.md](docs/testing-guide.md) V1~V6). **분리 이후 한 번도 실기동한 적이 없다.**
   Gemma-first 전환으로 티어 기본이 통째로 바뀌었으므로 검증 대상도 바뀌었다 —
   cpu-only/gpu-16gb 기본 경로(`runtime=server`)는 실행 이력이 0회다.
   여기서만 드러나는 미검증 부채: 카트리지 장착 런타임 반영 · docker 리네임 후 빌드·기동 ·
   PII 게이트 실동작 · **탈색된 기본 프롬프트의 응답 품질**(정적으로는 계약 보존만 확인했다).
2. **visibility 를 public 으로** — 사용자 승인 사항.
3. **직후 기본 브랜치 보호** — force push·삭제 금지 + CI 상태 검사 필수. 무료 플랜 private 에서는
   API 가 403 이라 **public 전환 뒤에만** 설정할 수 있다. 1인 리포이므로 PR 리뷰 필수화는 넣지 않는다
   (직접 push 경로가 막힌다).

> **민감어 스캔은 축을 넓게 잡을 것.** IP 대역·고유명사만이 아니라 호스트명·계정명·타 호스트
> 절대경로까지 본다. 참조되지 않는 죽은 코드도 스캔 대상이다.

## 작업 규칙

- **공개 전제 규율**: 내부 IP·호스트명·장비/사명 지칭·실운영 유래 데이터 커밋 금지. 커밋 전 `grep -riE "192\.168\." .` 0건 + 원 도메인·소속 관련 고유명사 스캔(목록은 리포 밖에서 관리) 통과 유지.
- **실측주의**: VRAM·tok/s·호환성 주장은 실행해본 수치만 기록. README 표·배지도 실측치만.
- **시크릿**: config.ini 실값·키·인증서 커밋 절대 금지(.gitignore 유지). knowledge 문서도 커밋 전 시크릿 패턴 스캔.
- **README 스타일 유지.** mermaid 라벨은 따옴표+`<br/>`, 괄호·`\n` 금지 — GitHub 렌더러가 깨진다. 복잡한 도식은 정적 SVG 커밋. **README.md(한국어)가 기본**, README.en.md(영문)는 항상 같은 내용으로 동기화한다.
- **문서도 영/한 쌍이다** (2026-07-31~). `docs/*.md`가 원본, `docs/*.en.md`가 영문판. **한국어를 고치면 같은 커밋에서 영문판도 고친다** — 한쪽만 갱신된 문서는 잘못된 문서보다 나쁘다(독자가 최신이라고 믿는다).
- **CI가 지키는 것**([.github/workflows/ci.yml](.github/workflows/ci.yml)): 문법 · `install.sh --dry-run` 완주 · 프리셋 파서 전 티어 · 카트리지 검증 · 시크릿 커밋 · 문서 내부 링크. **CI 통과는 "동작한다"가 아니다** — 엔진 종단은 [docs/testing-guide.md](docs/testing-guide.md) 실기동 검증 몫.
- **스크린샷·증적 이미지는 `docs/img/`에 PNG 고정**, 명명 `YYYY-MM-DD_환경_주제.png` (환경은 `t1-vm`·`t2-desktop`처럼 공개 가능한 표기). 참조 없는 이미지는 커밋하지 않는다.

## 기술 메모 (실측 근거)

- 원본 운영: Gemma-4-26B-A4B(MoE 활성 4B) UD-Q4_K_M 16GB, RTX 5090에서 ctx 524288·2슬롯 시 VRAM 28.4GB(그중 KV ~12GB). BGE-M3가 GPU 1.8GB 별도. 생성 221.2 tok/s·TTFT 16ms.
- 16GB GPU 실측(T2): gpt-oss-20b 13.2GB·178 tok/s / 26B MoE 오프로드 GPU 4.2GB·40.7 tok/s / 12B 7.8GB·99 tok/s.
- 16GB GPU 26B 레시피(검증된 경로): `n_ctx 32768` + `n_parallel 1` + `--n-cpu-moe <N>` → VRAM ~9GB, expert는 램 ~14GB 상주(램 32GB 필요).
- DB(MariaDB)는 대화 로깅 전용 — `[database] use_db_mode = False`로 완전 우회 가능. 필수 의존성은 Qdrant+BGE-M3뿐.
- 쿼리 검증 플러그인은 `[validation] plugin_module` 동적 로드(기본 off) — secops 전용 검증기는 이 리포에 포함하지 않는다.
- llama-server `--cache-reuse 256 --slot-prompt-similarity 0.5`(KV 캐시 재사용)는 gemma spawn만 승계, gpt-oss spawn엔 미적용(비대칭) — T2에서 A/B 후 기본값 결정.
