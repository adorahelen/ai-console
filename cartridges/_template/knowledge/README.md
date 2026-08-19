# knowledge/ — RAG 문서 형식

여기 놓인 YAML이 벌크 업로드(`POST /api/ai/prompts/bulk`)로 임베딩·적재된다.
한 파일 = 한 지식 단위. 형식은 두 종류:

## qna (지식 질의응답용)

```yaml
type: qna
name: my-topic-001            # 고유 식별자
enabled: true
question: 사용자가 물을 법한 질문 원형
answer: |
  검색되어 context로 주입될 답변 본문 (마크다운 가능)
aliases:                      # 같은 질문의 변형 표현 (검색 recall 향상)
- 질문 변형 1
- 질문 변형 2
```

## action (작업 생성용 — 쿼리·코드·설정 등)

```yaml
type: action
name: my-action-001
enabled: true
question: 이 작업을 요청하는 자연어 문장
cot: |
  - 생성 규칙·제약을 단계별로 기술 (모델이 따라갈 사고 과정)
answer: |
  기대 출력의 정답 예시 (JSON/코드/쿼리 등)
aliases: [...]
```

> 실전 감각: 잘 만든 qna는 "질문 원형 + aliases 변형"이 풍부하고, action은 cot에 생성 규칙이 단계별로 적혀 있다.
