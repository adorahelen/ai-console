# 기여 안내

ai-console에 기여해 줘서 고맙다. 이 문서는 **무엇을 받고 무엇을 받지 않는지**를 먼저 밝힌다 — 방향이 맞지 않는 PR에 시간을 쓰는 것이 서로에게 가장 아까운 일이라서다.

> English: see [In English](#in-english) at the bottom.

## 이 프로젝트의 방향 (바꾸지 않는 것)

이 네 가지는 설계 결정이지 미해결 과제가 아니다. 이걸 바꾸자는 PR은 코드보다 **이슈에서 먼저** 논의한다.

1. **아키텍처 유지** — FastAPI + llama-server(GGUF) + Qdrant/BGE-M3 2-way RRF + intent 분류 + 핸들러 게이트웨이. LangChain류 프레임워크 도입은 받지 않는다.
2. **카트리지는 기존 통로를 재사용한다** — 새 메커니즘을 발명하지 않는다. 프롬프트는 `config.ini [prompts]`, 지식은 `POST /api/ai/prompts/bulk`, 모델은 `models.yaml` 프리셋. 새 기능은 이 셋 위의 얇은 층으로 만든다.
3. **`models.yaml`이 모델의 단일 소스** — 프리셋 추가는 다운로드 명령과 **실측 VRAM** 없이는 받지 않는다.
4. **파인튜닝 없음** — 도메인 능력은 프롬프트와 RAG로만 만든다. 순정 가중치 원칙.

## 받는 기여

- **버그 수정** — 재현 절차가 있으면 가장 좋다.
- **카트리지** — 새 도메인의 프롬프트·지식 묶음. 아래 별도 절 참조.
- **설치·이식성** — 다른 배포판·다른 HW에서 `install.sh`가 깨지는 지점.
- **문서** — 특히 `docs/`의 영문화(현재 대부분 한국어).
- **모델 프리셋** — 실측치를 동반한 것.

## 받지 않는 기여

- 실측 없는 성능 주장. 이 리포는 **실행해 본 수치만** 기록한다 — VRAM·tok/s·호환성 전부.
- 대규모 리팩터링·포매팅 일괄 변경. 리뷰가 불가능해진다.
- 특정 벤더·상용 제품에 종속된 도메인 자산.
- 커밋에 포함된 시크릿·실제 인증서·내부 인프라 식별자(IP·호스트명·계정명). **이건 되돌리기 어렵다** — 아래 체크리스트를 반드시 통과시켜라.

## 시작하기

```bash
git clone https://github.com/adorahelen/ai-console-public.git ai-console
cd ai-console
./install.sh --dry-run        # 아무것도 설치하지 않고 전 단계를 출력만 한다
```

실제 설치와 HW 티어·프리셋 선택은 [README](README.md), 설치 경로의 세부는 [docs/install-paths.md](docs/install-paths.md)를 본다.

## PR 전에 돌릴 것

전부 로컬에서 몇 초 안에 끝난다. CI도 같은 것을 돈다.

```bash
# 1. 문법
python3 -m py_compile *.py
bash -n install.sh run.sh

# 2. 설치기 완주 (아무것도 설치·다운로드하지 않는다)
./install.sh --dry-run --yes --preset gemma4-12b-q4

# 3. 프리셋 파서 — install.sh가 venv 前 시스템 python으로 쓰는 경로라 stdlib만 쓴다
for t in cpu-only gpu-8gb gpu-16gb gpu-24gb-plus api; do
  python3 scripts/parse_preset.py "$t"
done

# 4. 카트리지를 건드렸다면
aibotctl cartridge validate cartridges/<이름>
```

**엔진 동작을 바꿨다면 실기동 검증이 필요하다.** 정적 검사로는 프롬프트 배선·RAG 종단·모델 응답 품질을 잡을 수 없다. 절차와 성공 기준은 [docs/testing-guide.md](docs/testing-guide.md)에 있고, PR 본문에 **무엇을 어디서 돌렸는지** 적어달라. 못 돌렸으면 못 돌렸다고 적으면 된다 — 검증하지 않은 것을 검증했다고 쓰는 쪽이 훨씬 나쁘다.

### 커밋 전 시크릿·식별자 체크리스트

```bash
git diff --cached | grep -inE '192\.168\.|10\.[0-9]+\.|api[_-]?key *= *[^ ]|BEGIN .*PRIVATE KEY'
```

- `config.ini` 실값, `api_keys/`, `ssl/` 은 `.gitignore` 대상이다 — `git add -f` 로 우회하지 마라.
- 내부 호스트명·개인 계정명·타 호스트 절대경로(`/home/<사내도메인>/...`)도 마찬가지다. 실제로 한 번 새어 들어온 적이 있다.
- 지식 YAML도 스캔한다. 문서에서 온 것이라도 그 안에 자격증명이 있을 수 있다.

## 카트리지 기여

카트리지는 코드가 아니라 **도메인 자산**이라 기준이 따로 있다.

1. `cartridges/_template/`을 복사해 시작한다. 형식은 [cartridge.yaml](cartridges/_template/cartridge.yaml)과 [knowledge/README.md](cartridges/_template/knowledge/README.md).
2. **실무 기본 단위는 지식 + `qna` 프롬프트다.** `system` 슬롯만 채운 단일형 카트리지는 기본 intent 모드에서 페르소나가 물리지 않는다(요청마다 `qna`/`action`/`plan`이 base를 덮는다).
3. `aibotctl cartridge validate`를 통과해야 한다 — stub(빈 항목)과 깨진 배선을 거른다.
4. **출처와 라이선스를 밝혀라.** 재배포 가능한 라이선스가 아닌 코퍼스에서 대량 변환한 지식은 받을 수 없다. LLM이 생성한 초안이 섞였다면 그것도 적는다.
5. 살아 있는 예시: [cartridges/console-guide/](cartridges/console-guide/) — 콘솔 자신의 사용법을 카트리지로 포장한 실물.

여러 포맷의 문서를 한꺼번에 굽는 배치 인제스터가 있다: `python ingest.py <소스_디렉터리> <카트리지_이름>`. 비정형 문서의 변환 결과는 **초안**이니 장착 전에 검수해라.

## 커밋과 PR

- **한 커밋에 한 가지 문제.** 리팩터링과 버그 수정을 섞지 않는다.
- 커밋 메시지는 `<타입>: <무엇을 왜>` — 기존 히스토리(`git log`)의 톤을 따르면 된다. 한국어·영어 모두 좋다.
- **무엇을 바꿨는지보다 왜 바꿨는지**를 적어라. diff는 이미 무엇을 말해 준다.
- 문서에 mermaid를 넣는다면: 라벨은 따옴표로 감싸고 줄바꿈은 `<br/>`, 괄호와 `\n`은 쓰지 마라 — GitHub 렌더러가 깨진다.

## 버그를 발견했다면

[이슈](https://github.com/adorahelen/ai-console-public/issues)를 열어라. 재현 절차·기대 동작·실제 동작, 그리고 **어떤 HW 티어와 프리셋**인지 적어주면 대부분의 왕복이 줄어든다.

보안 취약점은 공개 이슈로 열지 말고 리포 소유자에게 직접 연락해라.

## 라이선스

기여한 내용은 [MIT License](LICENSE) 하에 배포된다.

---

## In English

Thanks for considering a contribution. The short version:

**Fixed by design — discuss in an issue before writing code.** (1) The architecture stays: FastAPI + llama-server (GGUF) + Qdrant/BGE-M3 2-way RRF + intent classification + a handler gateway; no LangChain-style frameworks. (2) Cartridges reuse existing channels — prompts via `config.ini [prompts]`, knowledge via `POST /api/ai/prompts/bulk`, models via `models.yaml`; new features are thin layers over these. (3) `models.yaml` is the single source for models, and preset additions need **measured** VRAM. (4) No fine-tuning — domain ability comes from prompts and RAG only.

**Welcome:** bug fixes with repro steps, cartridges, install/portability fixes, documentation (especially translating `docs/` into English), model presets with measured numbers.

**Not accepted:** performance claims without measurements, sweeping refactors or formatting passes, vendor-locked domain assets, and anything containing secrets, real certificates, or internal infrastructure identifiers.

**Before opening a PR**, run the checks in [PR 전에 돌릴 것](#pr-전에-돌릴-것) — syntax (`py_compile`, `bash -n`), a full `./install.sh --dry-run` (installs and downloads nothing), the preset parser across all tiers, and `aibotctl cartridge validate` if you touched a cartridge. If you changed engine behavior, static checks are not enough: see [docs/testing-guide.md](docs/testing-guide.md) and state in the PR body **what you actually ran and where**. Saying "I could not run it" is fine; claiming verification you did not do is not.

**Cartridges** are domain assets, not code: state the source and license of your knowledge, disclose any LLM-generated drafts, and pass `aibotctl cartridge validate`. Note that a cartridge defining only the `system` slot will not take effect as a persona under the default intent mode — the practical unit is knowledge plus a `qna` prompt.

One commit per problem; explain **why**, not what. Contributions are released under the [MIT License](LICENSE).
