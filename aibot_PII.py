"""
aibot_PII.py — 온프레미스 PII 탐지/마스킹 모듈 (외부 통신 0).

3계층 탐지:
  1) 정규식 + 체크섬   — 한국 PII (주민/사업자/카드 등). 정확·고신뢰.
  2) NER (spaCy)       — 영문 인명/조직 등.
  3) 의미기반(임베딩)  — 정규식/NER이 놓친 자유서술 민감정보(한글 인명·주소·건강정보 등).
                          BGE-M3 등 임베딩 모델을 "주입"해서 사용. 모델 미주입 시 1+2만 동작.

임베딩 모델은 직접 import/로드 하지 않고 호출가능 객체(embedder)로 주입한다:
    embedder: Callable[[str], Sequence[float]]   # 텍스트 -> dense 벡터
모델은 새로 로드하지 않는다. RAG 파이프라인이 "이미 로드해 둔" 인스턴스를 받아
재사용한다:
    embedder = embedder_from_flagmodel(이미_로드된_BGEM3FlagModel)
    embedder = embedder_from_bge_adapter(이미_로드된_BGEEmbeddingAdapter)

공개 API:
    pii = AibotPII()                              # 1+2 계층
    pii = AibotPII(embedder=emb, enable_semantic=True)  # 1+2+3 계층
    pii.detect(text)              -> bool
    pii.analyze(text)             -> list[Finding]
    pii.mask(text)                -> str                  (<ENTITY> 치환)
    pii.mask_reversible(text)     -> (str, dict)          (게이트웨이용 토큰화 + 복원맵)
    pii.unmask(text, mapping)     -> str

의존성(전부 로컬):
    pip install presidio-analyzer presidio-anonymizer spacy numpy
    python -m spacy download en_core_web_lg
    # 의미기반 계층 사용 시: 프로젝트의 BGE-M3 (FlagEmbedding) 또는 임의 임베더
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
from presidio_analyzer.nlp_engine import NlpEngineProvider

EmbedderFn = Callable[[str], Sequence[float]]

# ─────────────────────────────────────────────────────────────────────────────
# 1) 한국 PII custom recognizers (체크섬 있는 항목은 validate_result 로 실검증)
# ─────────────────────────────────────────────────────────────────────────────


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# 한글-안전 경계: 숫자/한글이 같은 \w 라 \b 가 안 먹힘. 숫자 앞뒤만 막고 한글 인접은 허용.
_NB = r"(?<![\d-])"   # 앞: 숫자/하이픈 아님
_NA = r"(?![\d-])"    # 뒤: 숫자/하이픈 아님
_ND = r"(?<!\d)"
_NDA = r"(?!\d)"


class KrRrnRecognizer(PatternRecognizer):
    """주민등록번호 / 외국인등록번호 (YYMMDD-GXXXXXX). 유출방지 우선: 체크섬 실패해도 마스킹."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_RRN",
            patterns=[Pattern("rrn", _ND + r"\d{6}-?[1-8]\d{6}" + _NDA, 0.6)],
            context=["주민", "주민번호", "주민등록", "외국인등록"],
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        d = _digits(pattern_text)
        if len(d) != 13:
            return False
        mm, dd = int(d[2:4]), int(d[4:6])
        if not (1 <= mm <= 12 and 1 <= dd <= 31):
            return False
        w = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
        s = sum(int(d[i]) * w[i] for i in range(12))
        # 체크섬 통과=확신(True). 실패해도 RRN 형식이면 유출방지 위해 유지(None=기본score).
        return True if (11 - s % 11) % 10 == int(d[12]) else None


class KrBizRecognizer(PatternRecognizer):
    """사업자등록번호 (XXX-XX-XXXXX). 체크섬 실패해도 형식이면 마스킹(유출방지)."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_BIZ_NO",
            patterns=[Pattern("biz", _ND + r"\d{3}-?\d{2}-?\d{5}" + _NDA, 0.5)],
            context=["사업자", "사업자등록", "사업자번호"],
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        d = _digits(pattern_text)
        if len(d) != 10:
            return False
        w = [1, 3, 7, 1, 3, 7, 1, 3, 5]
        s = sum(int(d[i]) * w[i] for i in range(9)) + (int(d[8]) * 5) // 10
        return True if (10 - s % 10) % 10 == int(d[9]) else None


class KrCorpRecognizer(PatternRecognizer):
    """법인등록번호 (XXXXXX-XXXXXXX). 주민번호와 형식 겹쳐 score 낮게, context 의존."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_CORP_NO",
            patterns=[Pattern("corp", _ND + r"\d{6}-?\d{7}" + _NDA, 0.35)],
            context=["법인", "법인등록"],
        )


class KrPhoneRecognizer(PatternRecognizer):
    """휴대폰 + 유선전화."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_PHONE",
            patterns=[
                Pattern("mobile", _ND + r"01[016789]-?\d{3,4}-?\d{4}" + _NDA, 0.7),
                Pattern("landline", _ND + r"0(?:2|[3-6][1-5])-?\d{3,4}-?\d{4}" + _NDA, 0.55),
            ],
            context=["전화", "휴대폰", "핸드폰", "연락처", "tel"],
        )


class KrPassportRecognizer(PatternRecognizer):
    """여권번호 (영문 1자리 + 숫자 8자리)."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_PASSPORT",
            patterns=[Pattern("passport", r"(?<![A-Za-z0-9])[MSRODmsrod]\d{8}" + _NDA, 0.5)],
            context=["여권", "passport"],
        )


class KrDriverLicenseRecognizer(PatternRecognizer):
    """운전면허번호 (XX-XX-XXXXXX-XX)."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_DRIVER_LICENSE",
            patterns=[Pattern("driver", _ND + r"\d{2}-?\d{2}-?\d{6}-?\d{2}" + _NDA, 0.5)],
            context=["운전면허", "면허"],
        )


class KrCreditCardRecognizer(PatternRecognizer):
    """신용카드 (13~16자리 + Luhn 체크섬)."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_CREDIT_CARD",
            patterns=[Pattern("card", _NB + r"(?:\d{4}[- ]?){3}\d{1,4}" + _NDA, 0.4)],
            context=["카드", "신용카드", "card"],
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        d = _digits(pattern_text)
        if not (13 <= len(d) <= 16):
            return False
        total = 0
        for i, ch in enumerate(d[::-1]):
            n = int(ch)
            if i % 2 == 1:
                n = n * 2 - 9 if n * 2 > 9 else n * 2
            total += n
        # Luhn 통과=확신(True). 실패해도 카드 형식이면 마스킹 유지(유출 방지 우선, RRN 정책과 통일):
        # None 반환 시 결과가 패턴 score(0.4 ≥ threshold)로 유지되어 마스킹됨.
        return True if total % 10 == 0 else None


class KrAccountRecognizer(PatternRecognizer):
    """계좌번호 (포맷 가변 → context 의존, 오탐 줄이려 score 낮게)."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_ACCOUNT",
            patterns=[Pattern("account", _NB + r"\d{2,6}-\d{2,6}-\d{2,6}(?:-\d{1,6})?" + _NDA, 0.3)],
            context=["계좌", "계좌번호", "입금", "출금", "은행", "account"],
        )


class KrHealthInsuranceRecognizer(PatternRecognizer):
    """건강보험 가입자번호 (포맷 가변 → context 의존)."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_HEALTH_INSURANCE",
            patterns=[Pattern("hi", _ND + r"\d{1}-?\d{9,10}" + _NDA, 0.25)],
            context=["건강보험", "보험증", "가입자번호", "요양"],
        )


class KrVehicleRecognizer(PatternRecognizer):
    """차량번호판 (12가3456 / 123가4567)."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_VEHICLE",
            patterns=[Pattern("vehicle", _ND + r"\d{2,3}[가-힣]\s?\d{4}" + _NDA, 0.6)],
            context=["차량", "차량번호", "번호판", "자동차"],
        )


class KrIpRecognizer(PatternRecognizer):
    """IPv4/IPv6 (한글-안전). Presidio 내장 IP_ADDRESS 는 \\b 라 한글 인접 시 미탐 → 대체.
    IPv6 는 완전형(8그룹) 또는 :: 압축형만 매칭 → MAC(6그룹×2hex, :: 없음)과 충돌 안 함."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_IP",
            patterns=[
                Pattern("ipv4", r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", 0.6),
                Pattern(
                    "ipv6",
                    r"(?<![0-9A-Fa-f:])("
                    r"(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}|"
                    r"(?:[0-9A-Fa-f]{1,4}:){1,7}:|"
                    r"(?:[0-9A-Fa-f]{1,4}:){1,6}:[0-9A-Fa-f]{1,4}|"
                    r"(?:[0-9A-Fa-f]{1,4}:){1,5}(?::[0-9A-Fa-f]{1,4}){1,2}|"
                    r"(?:[0-9A-Fa-f]{1,4}:){1,4}(?::[0-9A-Fa-f]{1,4}){1,3}|"
                    r"(?:[0-9A-Fa-f]{1,4}:){1,3}(?::[0-9A-Fa-f]{1,4}){1,4}|"
                    r"(?:[0-9A-Fa-f]{1,4}:){1,2}(?::[0-9A-Fa-f]{1,4}){1,5}|"
                    r"[0-9A-Fa-f]{1,4}:(?::[0-9A-Fa-f]{1,4}){1,6}|"
                    r":(?:(?::[0-9A-Fa-f]{1,4}){1,7}|:)"
                    r")(?![0-9A-Fa-f:])",
                    0.7,
                ),
            ],
            context=["ip", "아이피", "접속", "주소"],
        )

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        if ":" in pattern_text:          # IPv6 — 패턴으로 충분
            return True
        return all(0 <= int(o) <= 255 for o in pattern_text.split("."))


class KrEmailRecognizer(PatternRecognizer):
    """이메일 (한글-안전)."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_EMAIL",
            patterns=[Pattern(
                "email",
                r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![A-Za-z])",
                0.9,
            )],
            context=["이메일", "메일", "email", "mail"],
        )


class KrMacRecognizer(PatternRecognizer):
    """MAC 주소 (한글-안전)."""

    def __init__(self):
        super().__init__(
            supported_entity="KR_MAC",
            patterns=[Pattern(
                "mac",
                r"(?<![0-9A-Fa-f:-])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f:-])",
                0.8,
            )],
        )


class CredentialRecognizer(PatternRecognizer):
    """자격증명 (api_key=, password:, token: 등 key=value 형태)."""

    def __init__(self):
        super().__init__(
            supported_entity="CREDENTIAL",
            patterns=[Pattern(
                "cred",
                r"(?i)(?:api[_-]?key|apikey|secret|access[_-]?key|"
                r"password|passwd|pwd|auth[_-]?token|token|bearer)"
                r"\s*[:=]\s*[\"']?[^\s\"']{4,}",
                0.85,
            )],
        )


class AwsKeyRecognizer(PatternRecognizer):
    """AWS Access Key ID (AKIA...)."""

    def __init__(self):
        super().__init__(
            supported_entity="AWS_KEY",
            patterns=[Pattern("aws", r"(?<![A-Za-z0-9])AKIA[0-9A-Z]{16}(?![A-Za-z0-9])", 0.9)],
        )


class JwtRecognizer(PatternRecognizer):
    """JWT 토큰 (eyJ....eyJ....sig)."""

    def __init__(self):
        super().__init__(
            supported_entity="JWT",
            patterns=[Pattern("jwt", r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", 0.9)],
        )


class PrivateKeyRecognizer(PatternRecognizer):
    """PEM 개인키 블록."""

    def __init__(self):
        super().__init__(
            supported_entity="PRIVATE_KEY",
            patterns=[Pattern(
                "pem",
                r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----[\s\S]+?-----END[^-]+-----",
                0.99,
            )],
        )


class ProviderTokenRecognizer(PatternRecognizer):
    """공급자 API 토큰/시크릿 (GitHub/Slack/Stripe/Google/OpenAI 등). 전부 한글-안전 경계."""

    def __init__(self):
        super().__init__(
            supported_entity="API_TOKEN",
            patterns=[
                # GitHub PAT/OAuth (ghp_/gho_/ghu_/ghs_/ghr_) + fine-grained(github_pat_)
                Pattern("github", r"(?<![A-Za-z0-9])gh[posru]_[A-Za-z0-9]{36,}(?![A-Za-z0-9])", 0.95),
                Pattern("github_pat", r"(?<![A-Za-z0-9])github_pat_[A-Za-z0-9_]{40,}(?![A-Za-z0-9])", 0.95),
                # Slack (xoxb-/xoxp-/xoxa-/xoxr-/xoxs-)
                Pattern("slack", r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{8,}(?![A-Za-z0-9])", 0.9),
                # Stripe 등 (sk_/pk_/rk_ + live|test)
                Pattern("stripe", r"(?<![A-Za-z0-9])(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{10,}(?![A-Za-z0-9])", 0.9),
                # Google API key (AIza...)
                Pattern("google", r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{30,}(?![A-Za-z0-9])", 0.9),
                # OpenAI (sk- / sk-proj-)
                Pattern("openai", r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9]{20,}(?![A-Za-z0-9])", 0.85),
            ],
        )


def custom_recognizers() -> List[PatternRecognizer]:
    """한국 PII + 네트워크 식별자 + 보안 자격증명 custom 인식기 (전부 한글-안전)."""
    return [
        # 한국 PII (구조화)
        KrRrnRecognizer(), KrBizRecognizer(), KrCorpRecognizer(),
        KrPhoneRecognizer(), KrPassportRecognizer(), KrDriverLicenseRecognizer(),
        KrCreditCardRecognizer(), KrAccountRecognizer(),
        KrHealthInsuranceRecognizer(), KrVehicleRecognizer(),
        # 네트워크/연락 식별자 (한글-안전 — 내장 \b 버전 대체)
        KrIpRecognizer(), KrEmailRecognizer(), KrMacRecognizer(),
        # 보안 자격증명 / 시크릿
        CredentialRecognizer(), AwsKeyRecognizer(), JwtRecognizer(), PrivateKeyRecognizer(),
        ProviderTokenRecognizer(),
    ]


# 사용 엔티티 = 정밀한 정규식+체크섬 detector 만 사용.
# Presidio 내장 NER(PERSON/LOCATION/ORGANIZATION/NRP/DATE_TIME/AGE 등)은 영어 system prompt 의
# 기술용어를 오탐 치환해 프롬프트를 손상시키므로 제외. 한글 인명/주소는 의미계층(별도 튜닝)에서 담당.
KR_ENTITIES = [
    "KR_RRN", "KR_BIZ_NO", "KR_CORP_NO", "KR_PHONE", "KR_PASSPORT",
    "KR_DRIVER_LICENSE", "KR_CREDIT_CARD", "KR_ACCOUNT",
    "KR_HEALTH_INSURANCE", "KR_VEHICLE", "KR_IP", "KR_EMAIL", "KR_MAC",
]
SECRET_ENTITIES = ["CREDENTIAL", "AWS_KEY", "JWT", "PRIVATE_KEY", "API_TOKEN"]
DEFAULT_ENTITIES: List[str] = KR_ENTITIES + SECRET_ENTITIES


@dataclass
class Finding:
    entity_type: str
    text: str
    start: int
    end: int
    score: float
    source: str = "pattern"  # pattern | ner | semantic


# ─────────────────────────────────────────────────────────────────────────────
# 3) 의미기반(임베딩) 인식기 — 정규식/NER이 놓친 자유서술 민감정보 탐지
# ─────────────────────────────────────────────────────────────────────────────

# 카테고리별 프로토타입 문장. 정규식/NER이 못 잡는 자유서술 민감정보를 의미기반으로 탐지.
# 개인정보보호법(PIPA) 민감정보 + 일반 식별정보를 폭넓게 커버. 운영시 도메인에 맞게 보강.
DEFAULT_SEMANTIC_PROTOTYPES: Dict[str, List[str]] = {
    # ── 일반 식별정보 (한글이라 NER 미탐) ──
    "KR_PERSON_NAME": [
        "제 이름은 김철수입니다", "담당자는 이영희 과장입니다", "고객 홍길동 님이 문의했습니다",
    ],
    "KR_ADDRESS": [
        "서울특별시 강남구 테헤란로 123", "주소는 경기도 성남시 분당구입니다", "자택은 부산 해운대구입니다",
    ],
    # ── PIPA 민감정보 (특별 보호 대상) ──
    "HEALTH_INFO": [
        "당뇨병 진단을 받았습니다", "환자의 병명은 고혈압입니다", "정신과 치료 이력이 있습니다",
        "암 수술을 받았습니다", "복용 중인 약은 인슐린입니다",
    ],
    "FINANCIAL_INFO": [
        "계좌 잔액은 천만원입니다", "급여 명세서상 연봉은", "대출 상환 내역", "신용등급이 낮습니다",
    ],
    "RELIGION_BELIEF": [
        "저는 기독교 신자입니다", "불교를 믿습니다", "특정 종교 활동에 참여합니다",
    ],
    "POLITICAL_VIEW": [
        "특정 정당을 지지합니다", "정치적 성향은 진보입니다", "노동조합에 가입했습니다",
    ],
    "SEXUAL_LIFE": [
        "성적 지향에 관한 내용", "성생활 관련 정보입니다",
    ],
    "RACE_ETHNICITY": [
        "인종은", "민족적 배경은", "출신 국가는",
    ],
    "CRIMINAL_RECORD": [
        "범죄 전과가 있습니다", "형사 처벌을 받은 이력", "수사를 받고 있습니다",
    ],
    "BIOMETRIC_GENETIC": [
        "지문 정보입니다", "홍채 인식 데이터", "유전자 검사 결과", "얼굴 인식 정보",
    ],
}


def _segment(text: str) -> List[Tuple[str, int, int]]:
    """텍스트를 문장/줄 단위로 분할하고 (조각, start, end) 반환."""
    segs: List[Tuple[str, int, int]] = []
    for m in re.finditer(r"[^\n。.!?！？]+", text):
        s = m.group().strip()
        if len(s) >= 2:
            start = m.start() + (len(m.group()) - len(m.group().lstrip()))
            segs.append((s, start, start + len(s)))
    return segs


class SemanticPIIRecognizer:
    """임베딩 코사인 유사도로 민감 카테고리를 탐지."""

    def __init__(
        self,
        embedder: EmbedderFn,
        prototypes: Optional[Dict[str, List[str]]] = None,
        threshold: float = 0.6,
    ):
        self.embedder = embedder
        self.threshold = threshold
        self.prototypes = prototypes or DEFAULT_SEMANTIC_PROTOTYPES
        self._labels: List[str] = []
        self._proto_matrix: Optional[np.ndarray] = None  # (n_proto, dim)

    @staticmethod
    def _norm(v: Sequence[float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float32)
        n = np.linalg.norm(a)
        return a / n if n > 0 else a

    def _ensure_prototypes(self):
        if self._proto_matrix is not None:
            return
        vecs, labels = [], []
        for label, phrases in self.prototypes.items():
            for p in phrases:
                vecs.append(self._norm(self.embedder(p)))
                labels.append(label)
        self._proto_matrix = np.vstack(vecs)
        self._labels = labels

    def analyze(self, text: str) -> List[Finding]:
        self._ensure_prototypes()
        findings: List[Finding] = []
        for seg, start, end in _segment(text):
            q = self._norm(self.embedder(seg))
            sims = self._proto_matrix @ q  # 정규화 벡터 → 코사인
            idx = int(np.argmax(sims))
            score = float(sims[idx])
            if score >= self.threshold:
                findings.append(
                    Finding(self._labels[idx], seg, start, end, score, source="semantic")
                )
        return findings


# ─────────────────────────────────────────────────────────────────────────────
# 메인 엔진
# ─────────────────────────────────────────────────────────────────────────────


class AibotPII:
    """온프레미스 PII 탐지/마스킹 엔진 (정규식 + NER + 선택적 의미기반)."""

    def __init__(
        self,
        entities: Optional[List[str]] = None,
        score_threshold: float = 0.35,
        spacy_model: str = "en_core_web_lg",
        language: str = "en",
        embedder: Optional[EmbedderFn] = None,
        enable_semantic: bool = False,
        semantic_threshold: float = 0.6,
        semantic_prototypes: Optional[Dict[str, List[str]]] = None,
        verbose: bool = False,
    ):
        self.entities = entities or DEFAULT_ENTITIES
        self.language = language
        self.verbose = verbose
        self._analyzer = self._build_analyzer(spacy_model, score_threshold)

        self._semantic: Optional[SemanticPIIRecognizer] = None
        if enable_semantic:
            if embedder is None:
                raise ValueError("enable_semantic=True 이면 embedder 를 주입해야 합니다.")
            self._semantic = SemanticPIIRecognizer(
                embedder, prototypes=semantic_prototypes, threshold=semantic_threshold
            )

    @staticmethod
    def _build_analyzer(spacy_model: str, score_threshold: float) -> AnalyzerEngine:
        import spacy

        if not spacy.util.is_package(spacy_model):
            raise RuntimeError(
                f"spaCy 모델 '{spacy_model}' 미설치. "
                f"`python -m spacy download {spacy_model}` 후 재시도."
            )
        nlp = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        ).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp, default_score_threshold=score_threshold)
        for rec in custom_recognizers():
            analyzer.registry.add_recognizer(rec)
        return analyzer

    # ── 탐지 ─────────────────────────────────────────────────────────────────
    _NER_ENTITIES = {"PERSON", "LOCATION", "ORGANIZATION", "NRP", "DATE_TIME", "AGE"}

    def _pattern_ner_findings(self, text: str) -> List[Finding]:
        results = self._analyzer.analyze(text=text, language=self.language, entities=self.entities)
        out = []
        for r in results:
            src = "ner" if r.entity_type in self._NER_ENTITIES else "pattern"
            out.append(Finding(r.entity_type, text[r.start:r.end], r.start, r.end, r.score, src))
        return out

    def analyze(self, text: str) -> List[Finding]:
        """모든 계층의 탐지를 병합. 겹치는 span 은 score 높은 것만 남긴다."""
        findings = self._pattern_ner_findings(text)
        if self._semantic is not None:
            findings += self._semantic.analyze(text)
        return self._resolve_overlaps(findings)

    @staticmethod
    def _resolve_overlaps(findings: List[Finding]) -> List[Finding]:
        # score 내림차순으로 greedy 선택, 이미 점유된 구간과 겹치면 제외
        chosen: List[Finding] = []
        occupied: List[Tuple[int, int]] = []
        for f in sorted(findings, key=lambda x: (-x.score, x.start)):
            if any(not (f.end <= s or f.start >= e) for s, e in occupied):
                continue
            chosen.append(f)
            occupied.append((f.start, f.end))
        return sorted(chosen, key=lambda x: x.start)

    def detect(self, text: str) -> bool:
        return bool(self.analyze(text))

    # ── 마스킹 ──────────────────────────────────────────────────────────────
    def _apply(self, text: str, findings: List[Finding], reversible: bool):
        mapping: Dict[str, str] = {}
        value_to_token: Dict[str, str] = {}
        counters: Dict[str, int] = {}
        out = text
        for f in sorted(findings, key=lambda x: x.start, reverse=True):
            if reversible:
                token = value_to_token.get(f.text)
                if token is None:
                    counters[f.entity_type] = counters.get(f.entity_type, 0) + 1
                    token = f"<{f.entity_type}_{counters[f.entity_type]}>"
                    value_to_token[f.text] = token
                    mapping[token] = f.text
                repl = token
            else:
                repl = f"<{f.entity_type}>"
            out = out[: f.start] + repl + out[f.end:]
        return (out, mapping) if reversible else out

    def _log_diff(self, tag: str, before: str, after: str, n_map: int):
        """치환 전/후 변화를 출력 (verbose=True 일 때)."""
        if not self.verbose:
            return
        print(f"\n===== [PII:{tag}] 치환 전(before) =====")
        print(before)
        print(f"----- [PII:{tag}] 치환 후(after) | 탐지 {n_map}건 -----")
        print(after)
        print("=" * 50)

    def mask(self, text: str) -> str:
        """탐지된 PII 를 <ENTITY> 로 치환 (비복원)."""
        out = self._apply(text, self.analyze(text), reversible=False)
        self._log_diff("mask", text, out, 0)
        return out

    def mask_reversible(self, text: str) -> Tuple[str, Dict[str, str]]:
        """PII 를 <ENTITY_n> 토큰으로 치환하고 복원맵 반환. 같은 값은 같은 토큰."""
        out, mapping = self._apply(text, self.analyze(text), reversible=True)
        self._log_diff("mask_reversible", text, out, len(mapping))
        return out, mapping

    @staticmethod
    def unmask(text: str, mapping: Dict[str, str]) -> str:
        if not mapping:
            return text
        # 1) 직접 치환
        for token, original in mapping.items():
            text = text.replace(token, original)
        # 2) LLM 이 마크다운 이스케이프(\_)나 공백을 끼워 변형한 토큰도 복원.
        #    역슬래시/공백을 무시하고 정규화 비교 (<KR\_IP\_1> → <KR_IP_1>).
        if "<" in text:
            norm = {re.sub(r"[\\\s]", "", k): v for k, v in mapping.items()}

            def _sub(m):
                key = re.sub(r"[\\\s]", "", m.group(0))
                return norm.get(key, m.group(0))

            text = re.sub(r"<[\\\sA-Za-z0-9_\-]+?>", _sub, text)
        return text

    # ── 게이트웨이: LLM 입력 전체(system 포함) 검수 ───────────────────────────
    def mask_chat(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
        """
        LLM 에 들어가는 모든 메시지(system/user/assistant 등)의 content 를
        한 번에 검수·마스킹한다. 전 메시지에 걸쳐 일관된 토큰을 쓰도록
        복원맵(mapping)을 공유하므로, 같은 PII 값은 어느 메시지에 있든 동일 토큰.

        Args:
            messages: [{"role": "system"|"user"|..., "content": str}, ...]
        Returns:
            (masked_messages, mapping)
            - masked_messages: content 가 토큰화된 동일 구조 리스트 (LLM 으로 전송)
            - mapping: 토큰→원문. LLM 응답에 unmask() 적용시 사용.
        """
        mapping: Dict[str, str] = {}
        value_to_token: Dict[str, str] = {}
        counters: Dict[str, int] = {}
        masked_messages: List[Dict[str, str]] = []

        for msg in messages:
            content = msg.get("content", "") or ""
            findings = self.analyze(content)
            out = content
            for f in sorted(findings, key=lambda x: x.start, reverse=True):
                token = value_to_token.get(f.text)
                if token is None:
                    counters[f.entity_type] = counters.get(f.entity_type, 0) + 1
                    token = f"<{f.entity_type}_{counters[f.entity_type]}>"
                    value_to_token[f.text] = token
                    mapping[token] = f.text
                out = out[: f.start] + token + out[f.end:]
            masked_messages.append({**msg, "content": out})
            self._log_diff(f"mask_chat:{msg.get('role','?')}", content, out, len(mapping))

        return masked_messages, mapping


# ─────────────────────────────────────────────────────────────────────────────
# 이미 로드된 임베딩 모델을 "재사용"해서 embedder 만들기 (모델 로딩 안 함)
#
# 중요: 이 모듈은 모델을 절대 새로 로드하지 않는다. RAG 파이프라인이 이미 들고
# 있는 인스턴스(아래 둘 중 무엇이든)를 받아 dense 벡터 추출 함수로 감쌀 뿐이다.
#   - BGEEmbeddingAdapter 인스턴스  → embedder_from_bge_adapter()
#   - BGEM3FlagModel 인스턴스        → embedder_from_flagmodel()
# ─────────────────────────────────────────────────────────────────────────────


def embedder_from_bge_adapter(adapter) -> EmbedderFn:
    """이미 로드된 BGEEmbeddingAdapter(RAG용)를 재사용. dense 벡터만 추출."""

    def embed(text: str) -> Sequence[float]:
        return adapter.generate_embedding(text)["data"][0]["embedding"]["dense"]

    return embed


def embedder_from_flagmodel(model) -> EmbedderFn:
    """이미 로드된 BGEM3FlagModel 인스턴스를 직접 재사용. dense 벡터만 뽑는 경량 경로
    (sparse/colbert 계산 생략 → PII 의미탐지에 충분하고 더 빠름)."""

    def embed(text: str) -> Sequence[float]:
        out = model.encode([text], return_dense=True,
                           return_sparse=False, return_colbert_vecs=False)
        return out["dense_vecs"][0]

    return embed


@lru_cache(maxsize=1)
def get_default_engine() -> AibotPII:
    """싱글톤 엔진 (정규식+NER, 모델 로딩 1회)."""
    return AibotPII()


# ─────────────────────────────────────────────────────────────────────────────
# 데모 / 자체 테스트
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    # ── 1+2 계층 (정규식 + NER) ──────────────────────────────────────────────
    pii = AibotPII()
    samples = [
        "가입자 주민번호 9001011234568, 사업자등록번호 1234567891, 휴대폰 010-1234-5678",
        "결제카드 4532-1234-5678-9014, 여권 M12345678, 이메일 user@corp.co.kr",
        "계좌 110-234-567890 (신한은행), 차량 12가3456, 접속 IP 192.168.1.25",
        "로그인 password=Secr3tP@ss, AWS키 AKIAIOSFODNN7EXAMPLE, MAC 00:1A:2B:3C:4D:5E",
        "[오탐] 잘못된주민번호 900101-1234560",
    ]
    print("=== [1+2계층] 정규식 + NER ===")
    for t in samples:
        print(f"\n원문 : {t}")
        for f in pii.analyze(t):
            print(f"   - {f.entity_type:18s} [{f.source}] score={f.score:.2f}  '{f.text}'")
        print(f"mask : {pii.mask(t)}")

    # ── 게이트웨이: LLM 입력 전체(system 포함) 검수 ───────────────────────────
    print("\n=== [게이트웨이] LLM 입력 전 모든 메시지 검수 (mask_chat) ===")
    chat = [
        {"role": "system", "content": "당신은 보안 분석가입니다. 관리자 연락처 010-1234-5678 로 보고하세요."},
        {"role": "user", "content": "김부장(주민 9001011234568)의 카드 4532-1234-5678-9014 가 유출됐어. 같은 연락처 010-1234-5678 확인해줘."},
    ]
    masked_chat, mapping = pii.mask_chat(chat)
    for m in masked_chat:
        print(f"   [{m['role']}] {m['content']}")
    print(f"   복원맵(입력기준 superset): {mapping}")
    # 입력엔 RRN·카드·전화가 있었지만, LLM 응답은 그중 '전화'만 언급할 수 있다.
    # unmask 는 출력에 실제 등장한 토큰만 복원 → 미언급 값은 그대로 출력에 안 나옴.
    sample_resp = "확인했습니다. 담당자에게 <KR_PHONE_1> 로 통보 완료했습니다."
    print(f"   LLM응답(토큰): {sample_resp}")
    print(f"   응답복원     : {pii.unmask(sample_resp, mapping)}")

    # ── 3계층 (의미기반) — 임베더 주입 통합 검증 ───────────────────────────────
    # 실제 BGE-M3 대신 'stub 임베더'로 배선(plumbing)만 검증한다.
    # 운영시: embedder = make_project_bge_embedder()  로 교체.
    if "--semantic-stub" in sys.argv:
        print("\n=== [3계층] 의미기반 (stub 임베더로 배선 검증) ===")
        rng = np.random.default_rng(0)
        _cache: Dict[str, np.ndarray] = {}

        def stub_embedder(text: str) -> Sequence[float]:
            # 결정적 가짜 임베딩: 같은 텍스트=같은 벡터. 의미품질 아님, 배선 검증용.
            if text not in _cache:
                h = abs(hash(text)) % (2**32)
                _cache[text] = np.random.default_rng(h).standard_normal(8)
            return _cache[text]

        # 프로토타입과 '동일 문자열' 세그먼트는 코사인 1.0 → 탐지되어야 함
        pii_sem = AibotPII(
            embedder=stub_embedder, enable_semantic=True, semantic_threshold=0.99,
            semantic_prototypes={"DEMO_NAME": ["제 이름은 김철수입니다"]},
        )
        t = "제 이름은 김철수입니다. 무관한 문장입니다."
        print(f"원문 : {t}")
        for f in pii_sem.analyze(t):
            print(f"   - {f.entity_type:18s} [{f.source}] score={f.score:.2f}  '{f.text}'")
        print(f"mask : {pii_sem.mask(t)}")
        print("\n※ 운영 전환: 이미 로드된 BGE-M3 를 재사용 (새 로딩 X)")
        print("   embedder = embedder_from_flagmodel(rag.model)   # 또는")
        print("   embedder = embedder_from_bge_adapter(rag.bge_adapter)")
        print("   pii = AibotPII(embedder=embedder, enable_semantic=True)")
