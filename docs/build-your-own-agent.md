# 자기만의 에이전트 만들기 — 처음부터 끝까지

> 🇬🇧 English: [build-your-own-agent.en.md](build-your-own-agent.en.md)

> 전제: [install.sh](../install.sh) 설치 완료 + Qdrant·콘솔 기동 상태.
> 아직이라면 → [README 빠른 시작](../README.md#-빠른-시작), 설치 상세는 [install-paths.md](install-paths.md).

도메인 에이전트를 만드는 길은 둘이다. **처음이라면 길 A(위저드)를 권한다.**

---

## 길 A — 온보딩 위저드 (브라우저, 파일 안 만짐)

```
브라우저 → https://localhost:8443/wizard
```

1. **캐릭터 생성** — 에이전트가 뭐 하는 존재인지 자연어로 서술한다 (예: "우리 회사 사내 규정 안내 도우미").
2. **지식 채우기** — 문서를 붙여넣거나 업로드한다.
3. 내부 포맷(intent 프롬프트·qna YAML) 변환은 **설치된 LLM이 대신한다** — 자기부트스트랩 원칙. 사용자는 YAML 스키마를 몰라도 된다.
4. **카트리지 저장 → 기동.**

설계 배경: [onboarding-design.md](onboarding-design.md)

---

## 길 B — 카트리지 수제작 (파일 기반, 전체 통제)

### 1. 템플릿 복사

```bash
cp -r cartridges/_template cartridges/my-domain
```

`cartridges/my-domain/cartridge.yaml`을 열어 이름·설명·권장 모델을 채운다. 이 파일 하나가 도메인의 매니페스트다.

### 2. 프롬프트 작성 — 도메인 지능의 본체

`prompts/` 하위에 최소 두 종류를 만든다:

- **intent 분류 프롬프트** — 사용자 질문을 어떤 의도로 나눌지. 원본 SecOps 카트리지는 `QNA`(지식 응답)·`ACTION`(쿼리 생성)·`PLAN`(다단계 계획)·`PLAYBOOK`(절차) 4종을 썼다. 내 도메인의 intent는 내가 설계한다 — 예를 들어 사내 규정 도우미라면 `QNA`·`FORM`(서식 안내) 2종이면 충분할 수 있다.
- **intent별 시스템 프롬프트** — 각 intent에서 모델이 따를 생성 규칙.

기본 형식은 `prompts/system/`의 기존 파일들(엔진 기본 프롬프트)을 참고해 같은 구조로 작성한다.

### 3. 지식 채우기 — RAG 문서

`knowledge/` 하위에 YAML을 만든다. 형식은 2종 — 자세한 스펙은 [\_template/knowledge/README.md](../cartridges/_template/knowledge/README.md):

- `qna` — 지식 질의응답. `question` + `answer` + `aliases`(질문 변형 — 검색 recall을 올리는 핵심).
- `action` — 작업 생성(쿼리·코드·설정 등). `question` + `cot`(생성 규칙) + `answer`(정답 예시).

### 4. 모델 선택

```bash
./install.sh --preset <키>     # models.yaml의 프리셋 키
```

이미 설치했다면 `config.ini [model] model=` 값과 llama-server 설정만 프리셋에 맞게 조정한다. 티어 가이드는 [models.yaml](../models.yaml) 주석과 [README 모델 표](../README.md#%EF%B8%8F-모델-선택--hw-티어별).

### 5. 장착 — config 연결 + 지식 업로드

**프롬프트 연결**: `config.ini [prompts]`에서 intent·qna·action 계열 키의 경로를 내 카트리지 파일로 바꾼다:

```ini
[prompts]
intent = ./cartridges/my-domain/prompts/intent_classification.yaml
action = ./cartridges/my-domain/prompts/action.yaml
...
```

**지식 업로드** (API 키는 `api_keys/` 디렉토리의 `*.key` 파일 첫 줄):

```bash
KEY=$(head -1 api_keys/*.key)
for f in cartridges/my-domain/knowledge/*.yaml; do
  curl -sk -X POST https://localhost:8443/api/ai/prompts/bulk \
    -H "Authorization: Bearer $KEY" \
    -F "files=@$f"
done
```

업로드된 YAML은 BGE-M3로 임베딩되어 Qdrant에 적재된다. 지식은 적재 즉시 물리고, 프롬프트 배선은
`aibotctl cartridge mount`(또는 위저드)가 런타임에 반영하므로 재기동이 필요 없다 — 반영이 확인 안 되면
`./run.sh restart`가 폴백이다.

### 6. 스모크 테스트

세 종류 질문으로 파이프라인 전체를 확인한다:

| 질문 | 기대 |
| :-- | :-- |
| 지식에 넣은 내용 그대로 질문 | 해당 intent로 분류 + 업로드한 지식이 context로 검색되어 답변에 반영 |
| 지식에 없는 도메인 질문 | intent는 맞게 분류, 근거 없음을 인정하는 답변 (환각 여부 확인) |
| 도메인 무관 잡담 | 기본/폴백 intent 처리 |

전 구간 검증 절차(설치→기동→인증→위저드→RAG 종단)는 [testing-guide.md](testing-guide.md).

---

## 살아있는 레퍼런스

[cartridges/console-guide/](../cartridges/console-guide/)가 이 문서의 전 과정을 담은 완성품이다 — 콘솔 자신의 사용법을 안내하는 카트리지로, cartridge.yaml·qna 프롬프트·지식 8건·장착 명령까지 그대로 베껴서 시작하면 된다.

## 팁

- **intent는 적게 시작한다.** 2종으로 시작해 분류가 흔들리는 질문 유형이 보일 때만 쪼갠다.
- **aliases에 투자하라.** RAG recall의 체감 품질은 질문 변형을 얼마나 채웠는지에 비례한다.
- **action의 `cot`가 품질을 좌우한다.** 생성 규칙을 단계별로 명시할수록 작은 모델에서도 출력이 안정된다.
- **지식 문서에 시크릿 금지.** 업로드 전 내부 호스트명·키·계정 패턴을 스캔하는 습관을 들인다.
