# console-guide — 예시 카트리지

ai-console **자신의 사용법을 안내하는 도우미**. "도메인 전체가 카트리지 하나"를 보여주는 최소 실물이다.
지식 8건은 전부 이 리포의 공개 문서에서 나왔다 — 새 카트리지를 만들 때 그대로 베껴서 시작하면 된다.

## 장착

```bash
# 1. qna 프롬프트 연결 (config.ini)
#    [prompts] 섹션에서:
#    qna = ./cartridges/console-guide/prompts/qna.yaml
#    (intent 분류·action·plan은 엔진 기본값 그대로)

# 2. 지식 업로드
KEY=$(head -1 api_keys/*.key)
for f in cartridges/console-guide/knowledge/*.yaml; do
  curl -sk -X POST https://localhost:8443/api/ai/prompts/bulk \
    -H "Authorization: Bearer $KEY" -F "files=@$f"
done

# 3. 콘솔 재기동 후 스모크 테스트
#    "설치 방법 알려줘" → 지식 기반 답변이 나오면 장착 성공
```

## 구조

```
console-guide/
├── cartridge.yaml       # 3슬롯 매니페스트
├── prompts/qna.yaml     # QNA 캐릭터 (grounding·스타일 규칙)
└── knowledge/*.yaml     # qna 형식 지식 8건 (설치·모델·카트리지·업로드·위저드·DB·인증·접속)
```
