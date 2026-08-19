#!/usr/bin/env python3
"""
BGE-M3 기반 RAG 모듈
aibot_rag_module.py와 동일한 인터페이스 제공하되 BGE-M3 임베딩 사용
"""

import os
import re
import pickle
import time
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
import numpy as np
warnings.filterwarnings("ignore", message=".*fast tokenizer.*pad method.*")
from FlagEmbedding import BGEM3FlagModel
import torch
import yaml
from config_utils import ConfigManager, qdrant_collection
from aibot_db_manager import AibotDBManager

# Qdrant 메모리 클라이언트 추가
try:
    from qdrant_client import QdrantClient, models
    QDRANT_SUPPORT = True
except ImportError:
    QDRANT_SUPPORT = False
    print("⚠️ Qdrant not available. Install with: pip install qdrant-client")

# -----------------------------
# Command/Function doc branding
# -----------------------------
_CMD_DOC_PAT = re.compile(
    r"^query-(?P<name>.+?)-(?P<kind>command|function)\.ya?ml$",
    re.IGNORECASE
)

# 쿼리에서 "하이픈이 포함된 토큰"을 잡는다 (공백 없는 명령어/함수명 매칭 목적)
# 예: esedb-columns, boxplot-command(쿼리에서 command 접미가 붙어올 수도 있어 허용)
_QUERY_HYPHEN_TOKEN = re.compile(r"\b[a-zA-Z][\w$]*(?:-[\w$]+)+\b")


def extract_mode_brand_from_filename(path_or_name: str) -> Optional[Dict[str, str]]:
    """
    파일명이 query-<name>-command.yaml 또는 query-<name>-function.yaml 인 경우
    - mode_brand: lp_command_ref 또는 lp_function_ref
    - command_name / func_name: <name> (하이픈 포함 그대로)
    를 반환한다.
    """
    if not path_or_name:
        return None

    base = os.path.basename(path_or_name).strip()
    m = _CMD_DOC_PAT.match(base)
    if not m:
        return None

    name = m.group("name").strip()
    kind = m.group("kind").lower()

    # 명령어/함수명 중간 공백은 허용하지 않음(요구사항). 있으면 제거하지 말고 그대로 둬서 오류를 드러내는 편이 낫다.
    # 필요 시 아래처럼 방어 가능:
    # if " " in name: return None

    if kind == "command":
        return {"mode_brand": "lp_command_ref", "command_name": name}
    else:
        return {"mode_brand": "lp_function_ref", "func_name": name}

_CMD_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,63}")  # 시작은 문자, 하이픈 허용, 길이 제한
_LOG_GUARD_RE = re.compile(
    r"\b(CEF:|LEEF:|DETECT\||%ASA-|SyslogAlertForwarder:|firewall|waf|ips|epp|mds|dvc=|src=|dst=)\b",
    re.IGNORECASE
)

# 의도(usage/command/function 등) 키워드
_INTENT_WORDS_RE = re.compile(
    r"\b("
    r"command|commands|usage|function|functions|syntax|option|options|parameter|parameters|arg|args|example|examples|"
    r"dsl|query|manual|reference|ref|"
    r"명령어|사용법|함수|문법|구문|옵션|파라미터|인자|예제|레퍼런스|참조"
    r")\b",
    re.IGNORECASE
)

# (추가) 로그/이벤트 포맷 신호: 여기 걸리면 intent가 있어도 우선 로그로 판단
_LOG_SIGNAL_RE = re.compile(
    r"\b(CEF:|LEEF:|DETECT\||SyslogAlertForwarder:|%ASA-|PulseSecure:|tlog:\s*\{|"
    r"src=|dst=|spt=|dpt=|dvc=|dvchost=|deviceExternalId=|rt=|proto=|act=|msg=)\b",
    re.IGNORECASE
)

def _is_short_query(q: str) -> bool:
    # “명령어만 던지는” 케이스를 살리기 위한 기준
    # 너무 넓히면 오탐 늘어남 -> 보수적으로
    return len(q) <= 28 and len(q.split()) <= 3

def looks_like_log_payload(text: str) -> bool:
    """
    '아래 로그...' 류 요청에서 붙는 원문 로그를 빠르게 감지해서,
    command/function 라우터를 타지 않도록 가드.
    """
    if not text:
        return False
    # 줄바꿈/파이프/키-값 등 로그 특징
    if "\n" in text:
        return True
    if text.count("|") >= 6:
        return True
    if _LOG_GUARD_RE.search(text):
        return True
    return False


def build_command_filter(tokens: List[str]) -> Optional["models.Filter"]:
    """
    Qdrant payload의 command_name 또는 func_name에 대해 MatchAny 필터 생성.
    """
    if not tokens:
        return None
    from qdrant_client import models

    # command_name / func_name 둘 중 어디로 들어갔는지 모르므로 should로 묶어준다.
    should = [
        models.FieldCondition(key="command_name", match=models.MatchAny(any=tokens)),
        models.FieldCondition(key="func_name", match=models.MatchAny(any=tokens)),
    ]
    return models.Filter(should=should)

_ID_PATTERNS = {
    "capec": re.compile(r"\bCAPEC[\s\-_:]*0*(\d{1,5})\b", re.IGNORECASE),
    "cwe":   re.compile(r"\bCWE[\s\-_:]*0*(\d{1,6})\b", re.IGNORECASE),
    "cve":   re.compile(r"\bCVE[\s\-_:]*((?:19|20)\d{2}[\-_:]*\d{3,7})\b", re.IGNORECASE),

    # ATT&CK (Technique/Sub-technique)
    "attack_t": re.compile(r"\bT(\d{4})(?:\.(\d{3}))?\b", re.IGNORECASE),
    # ATT&CK Groups / Software
    "attack_g": re.compile(r"\bG(\d{4})\b", re.IGNORECASE),
    "attack_s": re.compile(r"\bS(\d{4})\b", re.IGNORECASE),
}

def parse_security_ids(text: str) -> Dict[str, Any]:
    """
    CAPEC/CWE/CVE/ATT&CK 식별자를 텍스트에서 추출해 정규화.
    - 반환 예: {
        "capec_ids":[66], "cwe_ids":[79], "cve_ids":["CVE-2021-44228"],
        "attack_technique_ids":["T1059","T1059.003"], "attack_group_ids":["G0007"], "attack_software_ids":["S0002"],
        "norm_ids":["CAPEC-66","CWE-79",...]
      }
    """
    if not text:
        return {
            "capec_ids": [], "cwe_ids": [], "cve_ids": [],
            "attack_technique_ids": [], "attack_group_ids": [], "attack_software_ids": [],
            "norm_ids": []
        }

    capec_ids = sorted({int(m.group(1)) for m in _ID_PATTERNS["capec"].finditer(text)})
    cwe_ids   = sorted({int(m.group(1)) for m in _ID_PATTERNS["cwe"].finditer(text)})

    # CVE는 "CVE-YYYY-NNNN..."로 정규화
    cve_ids = set()
    for m in _ID_PATTERNS["cve"].finditer(text):
        s = m.group(1).replace("_", "-").replace(":", "-")
        # 숫자 구간 정리: "2021-44228" -> "CVE-2021-44228"
        parts = s.split("-")
        if len(parts) >= 2:
            cve_ids.add(f"CVE-{parts[0]}-{parts[1]}")
    cve_ids = sorted(cve_ids)

    # ATT&CK
    attack_technique_ids = set()
    for m in _ID_PATTERNS["attack_t"].finditer(text):
        base = f"T{m.group(1)}"
        sub  = m.group(2)
        attack_technique_ids.add(base)
        if sub:
            attack_technique_ids.add(f"{base}.{sub}")
    attack_technique_ids = sorted(attack_technique_ids)

    attack_group_ids = sorted({f"G{m.group(1)}" for m in _ID_PATTERNS["attack_g"].finditer(text)})
    attack_software_ids = sorted({f"S{m.group(1)}" for m in _ID_PATTERNS["attack_s"].finditer(text)})

    norm = []
    norm += [f"CAPEC-{i}" for i in capec_ids]
    norm += [f"CWE-{i}" for i in cwe_ids]
    norm += cve_ids
    norm += attack_technique_ids
    norm += attack_group_ids
    norm += attack_software_ids

    return {
        "capec_ids": capec_ids,
        "cwe_ids": cwe_ids,
        "cve_ids": cve_ids,
        "attack_technique_ids": attack_technique_ids,
        "attack_group_ids": attack_group_ids,
        "attack_software_ids": attack_software_ids,
        "norm_ids": norm
    }

def build_id_filter(ids: Dict[str, Any]) -> Optional["models.Filter"]:
    """
    Qdrant payload에 저장된 정규화 ID 필드(norm_ids 또는 capec_ids 등)에 대해 Filter 생성.
    - norm_ids 기반을 1순위로 사용(단일 배열로 관리가 쉬움)
    """
    from qdrant_client import models

    norm_ids = ids.get("norm_ids") or []
    if norm_ids:
        # payload.norm_ids: [ "CAPEC-66", "CWE-79", ... ] 같은 형태를 기대
        return models.Filter(
            must=[models.FieldCondition(
                key="norm_ids",
                match=models.MatchAny(any=norm_ids)
            )]
        )

    # norm_ids가 없으면 개별 필드로 시도(호환)
    should = []
    if ids.get("capec_ids"):
        should.append(models.FieldCondition(key="capec_ids", match=models.MatchAny(any=ids["capec_ids"])))
    if ids.get("cwe_ids"):
        should.append(models.FieldCondition(key="cwe_ids", match=models.MatchAny(any=ids["cwe_ids"])))
    if ids.get("cve_ids"):
        should.append(models.FieldCondition(key="cve_ids", match=models.MatchAny(any=ids["cve_ids"])))
    if ids.get("attack_technique_ids"):
        should.append(models.FieldCondition(key="attack_technique_ids", match=models.MatchAny(any=ids["attack_technique_ids"])))
    if ids.get("attack_group_ids"):
        should.append(models.FieldCondition(key="attack_group_ids", match=models.MatchAny(any=ids["attack_group_ids"])))
    if ids.get("attack_software_ids"):
        should.append(models.FieldCondition(key="attack_software_ids", match=models.MatchAny(any=ids["attack_software_ids"])))

    if should:
        return models.Filter(should=should)

    return None


class RAGSystemBGE:
    """BGE-M3 임베딩 기반 RAG 시스템"""

    def __init__(self,
                 file_extension: str = "yaml",
                 embedding_folder: str = "qa_embeddings",
                 use_db: bool = True,
                 sub_id: Optional[int] = None,
                 db_manager: Optional[AibotDBManager] = None,
                 config_manager = None):
        """
        BGE-M3 기반 RAG 시스템 초기화

        Args:
            file_extension: 문서 파일 확장자
            embedding_folder: 임베딩 저장 폴더
            use_db: DB 사용 여부
            sub_id: 구독 ID (None일 경우 기본값 1 사용)
            db_manager: DB 매니저 인스턴스
        """
        print("[BGE-M3 RAG 시스템 초기화 시작]")

        self.file_extension = file_extension
        self.embedding_folder = embedding_folder
        self.use_db = use_db
        # sub_id가 None이면 기본값 1 사용 (aibot_rag_module.py와 동일한 로직)
        self.sub_id = sub_id if sub_id is not None else 1
        self.db_manager = db_manager
        self.debug_qdrant_hybrid = False  # 디버그용 하이브리드 검색 로그 출력

        # 설정 관리자
        self.config_manager = config_manager or ConfigManager()

        # BGE-M3 모델
        self.bge_model = None

        # Qdrant 클라이언트 초기화
        self.qdrant_client = None
        self.use_qdrant_memory = QDRANT_SUPPORT

        # 임베딩/문서 데이터 캐시
        self.embeddings = {}

        # 문서 데이터 (구독별)
        self.documents = {}

        # 하이브리드 검색 가중치 (dense, sparse, colbert)
        self.default_weights = {
            'dense': 0.40,    # Dense 벡터 가중치
            'sparse': 0.30,   # Sparse 벡터 가중치
            'colbert': 0.30   # ColBERT 가중치
        }

        # 모델 및 임베딩 로드
        self._load_model()

        # Qdrant 서버 클라이언트 초기화 (메모리 모드 제거)
        self._init_qdrant_memory()

        # Qdrant 서버 데이터 상태만 체크 (로드하지 않음)
        if self.use_db and self.db_manager:
            self._check_qdrant_server_status()

        # Qdrant에서 command/function 이름 사전 구축
        self._build_command_name_sets()

    def _check_qdrant_server_status(self):
        """Qdrant 서버 상태만 체크 (실제 로드 없음)"""
        if not self.qdrant_client:
            print("❌ Qdrant 클라이언트가 초기화되지 않았습니다")
            return

        try:
            # 컬렉션 목록만 조회
            collections = self.qdrant_client.get_collections()
            bge_collections = [c.name for c in collections.collections if c.name == qdrant_collection()]

            if bge_collections:
                print(f"✅ Qdrant 서버에 BGE 컬렉션 발견")
                for collection_name in bge_collections:
                    try:
                        collection_info = self.qdrant_client.get_collection(collection_name)
                        # Qdrant API: vectors_count → points_count
                        vector_count = getattr(collection_info, 'points_count', getattr(collection_info, 'vectors_count', 0)) or 0
                        status = getattr(collection_info, 'status', None)
                        status_str = str(status).split('.')[-1] if status else 'unknown'
                        optimizer = getattr(collection_info, 'optimizer_status', None)
                        optimizer_str = str(getattr(optimizer, 'status', 'unknown')).split('.')[-1] if optimizer else 'unknown'
                        if vector_count > 0:
                            print(f"   📊 {collection_name}: {vector_count}개 벡터 | status: {status_str} | optimizer: {optimizer_str}")
                        else:
                            print(f"   ⚪ {collection_name}: 빈 컬렉션 (0개 벡터)")
                    except Exception as e:
                        print(f"   ❌ {collection_name}: 상태 확인 실패 - {str(e)}")
            else:
                print("⚪ Qdrant 서버에 BGE 컬렉션 없음")

        except Exception as e:
            print(f"❌ Qdrant 서버 상태 체크 실패: {e}")

    def _build_command_name_sets(self):
        """
        Qdrant payload에 저장된 mode_brand/command_name/func_name를 기반으로
        command/function 이름 사전을 구축한다.
        """
        self.command_name_set = set()
        self.func_name_set = set()

        if not (self.use_qdrant_memory and self.qdrant_client):
            return

        from qdrant_client import models

        # 모든 활성 sub_id를 이미 알고 있다면 그 리스트를 사용하세요.
        # 여기서는 현재 self.sub_id만 예시로 듭니다.
        sub_ids = [self.sub_id]

        for sub_id in sub_ids:
            # 단일 컬렉션 사용
            collection_name = qdrant_collection()

            # mode_brand가 command/function 레퍼런스 + sub_id 필터 (공통=1 또는 해당 구독)
            sub_id_filter = [
                models.FieldCondition(key="sub_id", match=models.MatchValue(value=1)),
            ]
            if sub_id != 1:
                sub_id_filter.append(
                    models.FieldCondition(key="sub_id", match=models.MatchValue(value=sub_id))
                )

            flt = models.Filter(
                must=[
                    models.Filter(should=sub_id_filter),  # sub_id 필터
                    models.Filter(should=[  # mode_brand 필터
                        models.FieldCondition(key="mode_brand", match=models.MatchValue(value="lp_command_ref")),
                        models.FieldCondition(key="mode_brand", match=models.MatchValue(value="lp_function_ref")),
                    ]),
                    models.FieldCondition(key="enabled", match=models.MatchValue(value=True)),  # enabled 필터
                ]
            )

            offset = None
            while True:
                try:
                    pts, offset = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=flt,
                        with_payload=True,
                        with_vectors=False,
                        limit=256,
                        offset=offset,
                    )

                    for p in pts:
                        payload = p.payload or {}
                        mb = payload.get("mode_brand")
                        if mb == "lp_command_ref":
                            name = payload.get("command_name")
                            if name:
                                self.command_name_set.add(str(name))
                        elif mb == "lp_function_ref":
                            name = payload.get("func_name")
                            if name:
                                self.func_name_set.add(str(name))

                    if offset is None:
                        break
                except Exception:
                    break  # 컬렉션이 없으면 건너뜀

        print(f"✅ command_name_set={len(self.command_name_set)} func_name_set={len(self.func_name_set)}")

    def reload_resources(self, sub_id=None):
        """
        리소스 리로드 (BGE 모드)
        - Qdrant는 서버 기반이라 벡터 리로드 불필요
        - command_name_set 캐시만 갱신
        """
        if sub_id:
            print(f"🔄 [BGE] 리소스 리로드 (sub_id: {sub_id})")
        else:
            print("🔄 [BGE] 리소스 리로드")

        # command/function 이름 캐시 갱신
        self._build_command_name_sets()
        print("✅ [BGE] 리소스 리로드 완료")

    def extract_command_like_tokens_strict(self, query: str) -> List[str]:
        """
        LLM이 COMMAND_REF로 판정한 후에만 호출됨.
        query에서 토큰 후보를 추출하고 Qdrant command/func 사전에 정확 일치하는 것만 반환.
        """
        if not query:
            return []

        q = query.strip()
        ql = q.lower()

        # # 1) 로그성 텍스트면 즉시 컷 — LLM이 이미 COMMAND_REF로 판정했으므로 불필요
        # if looks_like_log_payload(q) or _LOG_SIGNAL_RE.search(q) or ("아래 로그" in q) or ("로그에 적합한 앱" in q):
        #     return []

        # # 2) 의도 키워드 게이트 — LLM이 이미 판정했으므로 불필요
        # has_intent = bool(_INTENT_WORDS_RE.search(q))

        # 후보 추출
        cands = set()

        # 하이픈 포함 후보
        for m in _CMD_TOKEN_RE.finditer(q):
            cands.add(m.group(0))

        # 단일 토큰 후보 — LLM이 이미 판정했으므로 조건 없이 항상 추출
        # if _is_short_query(q) or has_intent:
        for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9_\-$]{1,32}\b", q):
            cands.add(m.group(0))

        if not cands:
            return []

        # # stop words 필터 — LLM이 이미 COMMAND_REF로 판정했으므로 불필요
        # stop = {
        #     "http", "https", "ssh", "tcp", "udp", "dns",
        #     "get", "post", "put", "delete", "options",
        #     "allow", "deny", "blocked", "event",
        # }
        # cands = {c for c in cands if c.lower() not in stop}

        print(f"[DEBUG cands] 토큰 후보: {cands}")

        # 사전 정확 매칭 (대소문자 무시)
        cmd_set = getattr(self, "command_name_set", set())
        fn_set  = getattr(self, "func_name_set", set())

        cmd_set_lower = {c.lower(): c for c in cmd_set}
        fn_set_lower = {f.lower(): f for f in fn_set}

        hits = []
        for c in cands:
            c_lower = c.lower()
            if c_lower in cmd_set_lower:
                hits.append(cmd_set_lower[c_lower])
            elif c_lower in fn_set_lower:
                hits.append(fn_set_lower[c_lower])

        print(f"[DEBUG extract_cmd] cands={list(cands)[:10]} cmd_set_size={len(cmd_set)} hits={hits}")

        if not hits:
            return []

        # # 의도 키워드 없으면 반환 안 함 — LLM이 이미 COMMAND_REF로 판정했으므로 불필요
        # if not has_intent:
        #     if not _is_short_query(q):
        #         return []
        #     if len(hits) > 2:
        #         return []

        return sorted(set(hits))
    
    def _load_model(self):
        """BGE-M3 모델 로드"""
        try:
            # GPU 사용 가능 여부 체크 (gpt 모드에서는 강제 CPU)
            default_model = self.config_manager.get_model_config().get('model', '').lower()
            if default_model == 'gpt':
                device = 'cpu'
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            use_fp16 = True if device == 'cuda' else False

            print(f"🔄 Loading BGE-M3 model on {device}...")

            # 모델 경로 (config.ini [embedding] bge_model_path)
            model_path = self.config_manager.config.get('embedding', 'bge_model_path', fallback='/data/models/bge-m3')

            if not os.path.exists(model_path):
                print(f"⚠️ Model path not found: {model_path}")
                print("Attempting to use default model location...")
                model_path = 'BAAI/bge-m3'  # HuggingFace 모델명

            self.bge_model = BGEM3FlagModel(
                model_path,
                use_fp16=use_fp16,
                device=device
            )

            print(f"✅ BGE-M3 model loaded successfully on {device.upper()}")

            if device == 'cuda':
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   📊 GPU Memory: {gpu_mem:.1f} GB")

        except Exception as e:
            print(f"❌ Failed to load BGE-M3 model: {e}")
            raise

    def _init_qdrant_memory(self):
        """Qdrant 클라이언트 초기화 (서버 또는 메모리 모드)"""
        try:
            if not QDRANT_SUPPORT:
                print("⚠️ Qdrant not available, falling back to numpy-based search")
                self.use_qdrant_memory = False
                return

            # Config에서 Qdrant 설정 읽기
            use_server = self.config_manager.config.get("qdrant", "use_server", fallback="False").lower() == "true"

            if use_server:
                # 서버 모드로 연결
                host = self.config_manager.config.get("qdrant", "host", fallback="localhost")
                port = int(self.config_manager.config.get("qdrant", "port", fallback="6333"))
                timeout = int(self.config_manager.config.get("qdrant", "timeout", fallback="10"))

                self.qdrant_client = QdrantClient(host=host, port=port, timeout=timeout)
                print(f"✅ Qdrant 서버 클라이언트 초기화 완료 ({host}:{port})")

                # 서버 연결 테스트
                try:
                    collections = self.qdrant_client.get_collections()
                    print(f"   📋 서버 연결 성공, 기존 컬렉션 수: {len(collections.collections)}")

                except Exception as conn_error:
                    print(f"⚠️ 서버 연결 실패, 메모리 모드로 폴백: {conn_error}")
                    self.qdrant_client = QdrantClient(":memory:")
                    print("✅ Qdrant 메모리 클라이언트로 초기화 완료")
            else:
                # 메모리 모드
                self.qdrant_client = QdrantClient(":memory:")
                print("✅ Qdrant 메모리 클라이언트 초기화 완료")

        except Exception as e:
            print(f"⚠️ Qdrant 초기화 실패, numpy 방식으로 폴백: {e}")
            self.use_qdrant_memory = False

    def _create_qdrant_collection(self, sub_id: int, doc_type: str = None):
        """Qdrant 컬렉션 생성 (단일 컬렉션 'bge' 사용: sub_id, doc_type은 payload에 저장)"""
        # 단일 컬렉션 사용
        collection_name = qdrant_collection()

        try:
            # 기존 컬렉션이 있으면 유지 (데이터 보존)
            try:
                info = self.qdrant_client.get_collection(collection_name)
                print(f"📁 기존 컬렉션 사용: {collection_name} (포인트: {info.points_count}개)")
                return collection_name
            except:
                pass  # 컬렉션 없음 → 아래에서 생성

            # 새 컬렉션 생성 (BGE-M3 전용 설정)
            DENSE_DIM = 1024    # BGE-M3 dense 차원
            COLBERT_DIM = 1024  # BGE-M3 colbert 차원

            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=DENSE_DIM,
                        distance=models.Distance.COSINE,
                    ),
                    "colbert": models.VectorParams(
                        size=COLBERT_DIM,
                        distance=models.Distance.COSINE,
                        multivector_config=models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM  # MaxSim
                        ),
                        hnsw_config=models.HnswConfigDiff(m=16),
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                },
            )
            return collection_name

        except Exception as e:
            print(f"❌ 컬렉션 생성 실패 {collection_name}: {e}")
            return None

    # ========================================
    # 검색 및 임베딩 관련 메서드 (Qdrant 서버 전용)
    # ========================================

    def generate_embeddings(self, text: str) -> Dict[str, Any]:
        """파일에서 임베딩 데이터 로드"""
        for doc_type in ['qna', 'action', 'plan']:
            embedding_path = f'/data/ai-agent/bge_m3_embeddings_{doc_type}.pkl'

            if os.path.exists(embedding_path):
                try:
                    print(f"📂 Loading {doc_type} embeddings from {embedding_path}...")
                    with open(embedding_path, 'rb') as f:
                        self.embeddings[doc_type] = pickle.load(f)

                    # 문서 정보 추출
                    if 'documents' in self.embeddings[doc_type]:
                        for doc in self.embeddings[doc_type]['documents']:
                            file_path = doc.get('file_path', '')
                            self.documents[doc_type][file_path] = doc

                    num_docs = len(self.embeddings[doc_type].get('dense_vecs', []))
                    print(f"✅ Loaded {num_docs} {doc_type} embeddings")

                except Exception as e:
                    print(f"⚠️ Failed to load {doc_type} embeddings: {e}")
            else:
                print(f"⚠️ {doc_type} embeddings not found at {embedding_path}")

    def _load_embeddings_from_db(self):
        """BGE 임베딩 데이터 상태 체크 (Qdrant 서버 우선)"""
        if not self.db_manager:
            print("❌ DB 매니저가 설정되지 않았습니다")
            return

        print("🔍 BGE 데이터 상태 체크 시작...")

        # Qdrant 서버가 활성화된 경우 - 이것만 사용!
        if self.use_bge_vector_db and self.bge_vector_manager and self.bge_vector_manager.qdrant_enabled:
            print("🚀 Qdrant 서버 모드 - MariaDB/FAISS 체크 생략")
            return self._check_qdrant_server_data()

        # Qdrant 서버가 없는 경우에만 레거시 방식
        print("⚠️ Qdrant 서버 비활성화 - 레거시 모드로 실행")
        if self.use_bge_vector_db and self.bge_vector_manager:
            return self._load_from_bge_vector_db()

        # 최후 수단: ai_prompts 테이블 직접 로드
        print("🔄 기존 방식: ai_prompts 테이블에서 BGE 데이터 로드...")

        # 1. 모든 활성 구독 조회 (기존 시스템과 동일)
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id, name FROM ai_subscriptions WHERE expires_at IS NULL OR expires_at > NOW()")
                    subscriptions = cursor.fetchall()
                    sub_ids = [row['id'] if isinstance(row, dict) else row[0] for row in subscriptions]
                    print(f"📋 활성 구독: {len(sub_ids)}개 - {sub_ids}")
        except Exception as e:
            print(f"❌ 구독 목록 조회 실패: {e}")
            return

        # 2. 각 구독별로 BGE 데이터 로드
        loaded_count = 0
        total_bge_count = 0

        for sub_id in sub_ids:
            print(f"\n{'='*60}")
            print(f"📦 구독 ID {sub_id} BGE 데이터 로드 중...")
            print(f"{'='*60}")

            try:
                bge_count = self._load_subscription_bge_data(sub_id)
                if bge_count > 0:
                    loaded_count += 1
                    total_bge_count += bge_count
                    print(f"✅ 구독 {sub_id}: {bge_count}개 BGE 임베딩 로드됨")
                else:
                    print(f"⚠️ 구독 {sub_id}: BGE 데이터 없음")
            except Exception as e:
                print(f"❌ 구독 {sub_id} 로드 실패: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n✅ BGE 데이터 로드 완료: {loaded_count}/{len(sub_ids)}개 구독")
        print(f"📊 총 {total_bge_count}개의 BGE 임베딩 처리됨")

    def _check_qdrant_server_data(self):
        """Qdrant 서버에 있는 데이터 개수만 체크 (단일 컬렉션 방식)"""
        if not self.bge_vector_manager or not self.bge_vector_manager.qdrant_enabled:
            print("❌ Qdrant 서버가 활성화되지 않음")
            return

        try:
            from qdrant_client import models

            # 1. 모든 활성 구독 조회
            connection = self.db_manager.get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT id, name FROM ai_subscriptions WHERE expires_at IS NULL OR expires_at > NOW()")
            subscriptions = cursor.fetchall()
            sub_ids = [row['id'] if isinstance(row, dict) else row[0] for row in subscriptions]
            print(f"📋 활성 구독: {len(sub_ids)}개 - {sub_ids}")
            cursor.close()
            connection.close()

            # 2. 단일 컬렉션(bge) 데이터 개수 확인
            collection_name = qdrant_collection()
            total_vectors = 0

            try:
                # 컬렉션 존재 여부 및 벡터 개수 확인
                collection_info = self.bge_vector_manager.qdrant_client.get_collection(collection_name)
                total_vectors = collection_info.points_count or 0
                print(f"   📊 {collection_name}: {total_vectors}개 벡터")

                # doc_type별 개수 확인
                for doc_type in ['qna', 'action', 'plan']:
                    try:
                        count_result = self.bge_vector_manager.qdrant_client.count(
                            collection_name=collection_name,
                            count_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="doc_type",
                                        match=models.MatchValue(value=doc_type)
                                    )
                                ]
                            )
                        )
                        doc_count = count_result.count
                        if doc_count > 0:
                            print(f"      └ {doc_type}: {doc_count}개")
                    except:
                        pass

                # 구독별 데이터 개수 확인
                for sub_id in sub_ids:
                    try:
                        count_result = self.bge_vector_manager.qdrant_client.count(
                            collection_name=collection_name,
                            count_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="sub_id",
                                        match=models.MatchValue(value=sub_id)
                                    )
                                ]
                            )
                        )
                        sub_count = count_result.count
                        if sub_count > 0:
                            print(f"      └ sub_id={sub_id}: {sub_count}개")
                    except:
                        pass

            except Exception as e:
                print(f"   ⚪ {collection_name}: 없음 - {e}")

            print(f"\n✅ Qdrant 서버 데이터 체크 완료")
            print(f"📊 단일 컬렉션 'bge', {total_vectors}개 벡터")
            print("🚀 실제 벡터 로드는 검색 시점에 수행됩니다")

        except Exception as e:
            print(f"❌ Qdrant 서버 데이터 체크 실패: {e}")
            import traceback
            traceback.print_exc()

    def _load_from_bge_vector_db(self):
        """BGE 벡터 DB 상태 체크 (실제 로드는 하지 않음)"""
        if not self.bge_vector_manager:
            print("❌ BGE 벡터 DB 매니저가 없습니다")
            return

        try:
            # 1. 모든 활성 구독 조회
            connection = self.db_manager.get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT id, name FROM ai_subscriptions WHERE expires_at IS NULL OR expires_at > NOW()")
            subscriptions = cursor.fetchall()
            sub_ids = [row['id'] if isinstance(row, dict) else row[0] for row in subscriptions]
            print(f"📋 활성 구독: {len(sub_ids)}개 - {sub_ids}")
            cursor.close()
            connection.close()

            # 2. FAISS 인덱스 존재 여부만 체크 (로드하지 않음)
            print("\n🔍 BGE FAISS 인덱스 상태 체크...")
            available_indices = 0

            for sub_id in sub_ids:
                indices_found = 0
                for doc_type in ['qna', 'action', 'plan']:
                    if self.bge_vector_manager._check_faiss_index_exists(sub_id, doc_type):
                        indices_found += 1

                if indices_found > 0:
                    print(f"✅ 구독 {sub_id}: {indices_found}개 FAISS 인덱스 사용 가능")
                    available_indices += indices_found
                else:
                    print(f"⚪ 구독 {sub_id}: FAISS 인덱스 없음")

            print(f"\n✅ BGE 벡터 DB 상태 체크 완료")
            print(f"📊 총 {len(sub_ids)}개 구독, {available_indices}개 FAISS 인덱스 사용 가능")
            print("🚀 실제 인덱스 로드는 검색 시점에 수행됩니다")

        except Exception as e:
            print(f"❌ BGE 벡터 DB 상태 체크 실패: {e}")
            import traceback
            traceback.print_exc()

    def _load_subscription_bge_data(self, sub_id: int) -> int:
        """구독 BGE 데이터 상태 체크 (실제 로드 없음)"""
        print(f"ℹ️ 구독 {sub_id}: Qdrant 서버 모드 - 대량 로드 건너뛰기")
        return 0  # Qdrant 서버에서 동적 로드

    def _load_subscription_to_qdrant(self, sub_id: int) -> int:
        """특정 구독의 BGE 데이터를 Qdrant 메모리 컬렉션에 로드"""

        import pickle
        import yaml
        import numpy as np
        from qdrant_client import models

        # -----------------------------
        # 0) 프롬프트 조회
        # -----------------------------
        try:
            all_prompts = self.db_manager.get_prompts(s_id=sub_id, model="gpt-oss")
            print(f"📊 구독 {sub_id} 프롬프트 데이터: {len(all_prompts)}개")
        except Exception as e:
            print(f"❌ 구독 {sub_id} 프롬프트 조회 실패: {e}")
            return 0

        # 문서 메타데이터 저장용
        if sub_id not in self.documents:
            self.documents[sub_id] = {}
        for doc_type in ["qna", "action", "plan"]:
            self.documents[sub_id][doc_type] = {}

        # -----------------------------
        # 1) 컬렉션 생성
        # -----------------------------
        collections = {}
        for doc_type in ["qna", "action", "plan"]:
            collection_name = self._create_qdrant_collection(sub_id, doc_type)
            if collection_name:
                collections[doc_type] = collection_name

        # -----------------------------
        # 1-1) (중요) sparse가 “진짜로” 컬렉션에 등록되어 있는지 확인
        #      - 아래 출력에서 sparse_vectors_config(또는 sparse_vectors)이 존재하는지 확인하세요.
        #      - 이름이 using="sparse"와 동일하게 "sparse"로 되어 있어야 검색에서 동작합니다.
        # -----------------------------
        for doc_type, collection_name in collections.items():
            try:
                info = self.qdrant_client.get_collection(collection_name)
                # print(f"🧾 컬렉션 정보({collection_name}) 확인:")
                # 아래를 보고 확인할 것:
                #   1) dense/colbert는 vectors_config에 있는지
                #   2) sparse는 sparse_vectors_config에 있는지
                #   3) sparse 이름이 정확히 "sparse"인지
                # print(info)
            except Exception as e:
                print(f"⚠️ 컬렉션 정보 조회 실패({collection_name}): {e}")

        # BGE 데이터 처리 및 Qdrant 업로드
        bge_count = 0
        points_by_type = {"qna": [], "action": [], "plan": []}

        # sparse 통계 집계(디버깅용)
        # - sparse가 “전혀 동작 안 한다”면 대부분 아래 값이 0이거나(빈 벡터),
        #   max가 매우 작게 나오거나, 혹은 컬렉션 스키마에 sparse 등록이 누락된 경우입니다.
        sparse_stats = {
            "qna": {"docs": 0, "empty": 0, "nz_total": 0, "max_total": 0.0, "nz_max": 0},
            "action": {"docs": 0, "empty": 0, "nz_total": 0, "max_total": 0.0, "nz_max": 0},
            "plan": {"docs": 0, "empty": 0, "nz_total": 0, "max_total": 0.0, "nz_max": 0},
        }

        for row in all_prompts:
            try:
                name = row.get("name", "")
                source = row.get("source", "")

                vector_data = row.get("vector")
                if not vector_data:
                    continue

                # -----------------------------
                # 2) BGE 벡터 역직렬화 및 타입 확인
                # -----------------------------
                try:
                    if isinstance(vector_data, (bytes, bytearray)):
                        vector_obj = pickle.loads(vector_data)
                    else:
                        continue

                    # 확인 포인트:
                    # - vector_obj["type"] == "bge_m3" 인지
                    # - dense/sparse/colbert 키가 다 존재하는지
                    if not (isinstance(vector_obj, dict) and vector_obj.get("type") == "bge_m3"):
                        continue
                    if not all(k in vector_obj for k in ("dense", "sparse", "colbert")):
                        print(f"⚠️ 벡터 키 누락: {name} (keys={list(vector_obj.keys())})")
                        continue

                except (pickle.UnpicklingError, TypeError, KeyError) as e:
                    print(f"⚠️ 벡터 역직렬화 실패: {name}: {e}")
                    continue

                # -----------------------------
                # 3) YAML 컨텐츠 처리 + 메타 추출
                # -----------------------------
                yaml_content = row.get("prompt", "").replace("¶", "\n")
                if not yaml_content.startswith("---"):
                    yaml_content = "---\n" + yaml_content

                filename = (source or name).split("/")[-1]
                metadata = {
                    "name": row.get("name", ""),
                    "type": row.get("type", "qna"),
                    "filename": filename,
                    "path": source or name,
                    "description": row.get("description", ""),
                    "content": yaml_content,
                }

                if "---" in yaml_content and len(yaml_content) > 10:
                    try:
                        parsed_yaml = yaml.safe_load(yaml_content)
                        if isinstance(parsed_yaml, dict):
                            metadata.update(
                                {
                                    "name": parsed_yaml.get("name", metadata["name"]),
                                    "type": parsed_yaml.get("type", metadata["type"]),
                                    "description": parsed_yaml.get("description", metadata["description"]),
                                }
                            )
                    except Exception:
                        pass

                doc_type = (metadata["type"] or "qna").lower()
                if doc_type not in ["qna", "action", "plan"]:
                    doc_type = "qna"

                # -----------------------------
                # 4) Dense 변환
                # -----------------------------
                dense_vec = vector_obj["dense"]
                if hasattr(dense_vec, "tolist"):
                    dense_vec = dense_vec.tolist()
                elif isinstance(dense_vec, np.ndarray):
                    dense_vec = dense_vec.astype(float).tolist()
                else:
                    dense_vec = list(dense_vec)

                # -----------------------------
                # 5) Sparse 변환 + (중요) sparse 통계/로그
                # -----------------------------
                sparse_dict = vector_obj["sparse"]

                # BGE sparse 형식: {index: value} 형태가 기대값
                if isinstance(sparse_dict, dict):
                    sparse_indices, sparse_values = [], []
                    for idx, val in sparse_dict.items():
                        sparse_indices.append(int(idx))
                        if hasattr(val, "item"):
                            sparse_values.append(float(val.item()))
                        else:
                            sparse_values.append(float(val))

                    # (권장) 인덱스 정렬: 디버깅/일관성에 도움
                    if sparse_indices:
                        pairs = sorted(zip(sparse_indices, sparse_values), key=lambda x: x[0])
                        sparse_indices, sparse_values = map(list, zip(*pairs))
                    else:
                        sparse_indices, sparse_values = [], []

                    sparse_vec = models.SparseVector(indices=sparse_indices, values=sparse_values)

                elif isinstance(sparse_dict, models.SparseVector):
                    sparse_vec = sparse_dict
                else:
                    # 예상치 못한 형식이면 로그 남기고 건너뜀
                    print(f"⚠️ sparse 형식이 예상과 다름: {name} type={type(sparse_dict)}")
                    continue

                # --- sparse 디버깅 통계 업데이트 ---
                nz = len(getattr(sparse_vec, "indices", []) or [])
                mx = max(getattr(sparse_vec, "values", []) or [0.0]) if nz else 0.0

                st = sparse_stats[doc_type]
                st["docs"] += 1
                if nz == 0:
                    st["empty"] += 1
                st["nz_total"] += nz
                st["nz_max"] = max(st["nz_max"], nz)
                st["max_total"] += mx  # max값을 합산해 평균 추정
                # 개별 문서에 대해 너무 많이 찍으면 로그가 폭주하므로, 드물게 샘플만 출력
                '''if st["docs"] <= 3 or (st["docs"] % 500 == 0):
                    print(f"   🔎 [SPARSE DOC 샘플] type={doc_type} docs={st['docs']} nz={nz} max={mx:.6f} name={metadata['name']}")'''

                # -----------------------------
                # 6) ColBERT 변환 (multivector)
                # -----------------------------
                colbert_vec = vector_obj["colbert"]
                if hasattr(colbert_vec, "tolist"):
                    colbert_matrix = colbert_vec.tolist()
                elif isinstance(colbert_vec, np.ndarray):
                    colbert_matrix = colbert_vec.astype(float).tolist()
                else:
                    colbert_matrix = list(colbert_vec)

                # 1D면 2D로 보정
                if (
                    isinstance(colbert_matrix, list)
                    and len(colbert_matrix) > 0
                    and not isinstance(colbert_matrix[0], list)
                ):
                    colbert_matrix = [colbert_matrix]

                # -----------------------------
                # 7) Point 생성
                # -----------------------------
                point_id = len(points_by_type[doc_type])

                payload = {
                    "file_path": f"{doc_type}/{metadata['name']}",
                    "original_file_path": metadata["name"],
                    "content": yaml_content,
                    "description": metadata["description"],
                    "doc_type": doc_type,
                    "sub_id": sub_id,
                }
                
                # ✅ command/function 문서면 mode_brand 및 이름 필드 주입 (파일명 기반)
                brand = extract_mode_brand_from_filename(metadata.get("filename") or metadata.get("name") or metadata.get("path"))
                if brand:
                    payload.update(brand)

                # ✅ ID 추출: file_path + content + description를 합쳐서 파싱(명칭/본문 어디에 있든 잡음)
                id_ctx = f"{payload.get('path', '')}\n{payload.get('description','')}\n{payload.get('content','')}"
                ids = parse_security_ids(id_ctx)

                # ✅ payload에 필드 주입
                payload.update(ids)

                point = models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "colbert": colbert_matrix,
                        "sparse": sparse_vec,
                    },
                    payload=payload,
                )
                points_by_type[doc_type].append(point)

                self.documents[sub_id][doc_type][metadata["name"]] = {
                    "file_path": f"{doc_type}/{metadata['name']}",
                    "original_file_path": metadata["name"],
                    "content": yaml_content,
                    "description": metadata["description"],
                }

                bge_count += 1

            except Exception as e:
                print(f"⚠️ 임베딩 처리 실패 ({row.get('name', 'Unknown')}): {e}")
                continue

        # -----------------------------
        # 8) 업로드 전 sparse 통계 요약 출력 (핵심 체크)
        # -----------------------------
        # 확인할 것:
        # - empty 비율이 높으면(예: 30% 이상) sparse 자체가 비어있는 문서가 많아 검색이 약해집니다.
        # - nz_avg가 극단적으로 작으면(예: 평균 0~5) sparse가 사실상 거의 없다는 의미입니다.
        # - mx_avg가 지나치게 작으면 값 스케일이 붕괴된 것일 수 있습니다.
        '''print("📈 [SPARSE 통계 요약] (업로드 직전)")
        for t, st in sparse_stats.items():
            if st["docs"] == 0:
                print(f" - {t}: docs=0")
                continue
            nz_avg = st["nz_total"] / st["docs"]
            mx_avg = st["max_total"] / st["docs"]
            empty_ratio = (st["empty"] / st["docs"]) * 100.0
            print(
                f" - {t}: docs={st['docs']}, empty={st['empty']}({empty_ratio:.1f}%), "
                f"nz_avg={nz_avg:.1f}, nz_max={st['nz_max']}, mx_avg={mx_avg:.6f}"
            )'''

        # -----------------------------
        # 9) Qdrant 업로드
        # -----------------------------
        for doc_type, points in points_by_type.items():
            if points and doc_type in collections:
                try:
                    self.qdrant_client.upsert(
                        collection_name=collections[doc_type],
                        points=points,
                    )
                    print(f"✅ {doc_type}: {len(points)}개 문서가 Qdrant 컬렉션에 업로드됨")
                except Exception as e:
                    print(f"❌ {doc_type} Qdrant 업로드 실패: {e}")
        '''
        # ✅ 업로드 후 컬렉션 상태 재확인 (points_count, indexed_vectors_count 확인)
        for doc_type, collection_name in collections.items():
            try:
                info_after = self.qdrant_client.get_collection(collection_name)
                print(f"🧾 [업로드 후] 컬렉션 정보({collection_name}) 확인:")
                # 확인 포인트:
                # - points_count가 0이 아닌지
                # - sparse_vectors={'sparse': ...}가 유지되는지
                # - indexed_vectors_count는 최적화/인덱싱 타이밍에 따라 바로 안 늘 수 있음(지연 가능)
                print(info_after)
            except Exception as e:
                print(f"⚠️ [업로드 후] 컬렉션 정보 조회 실패({collection_name}): {e}")

        # ✅ 업로드 후: sparse-only / dense-only 결과가 나오는지 스모크 테스트
        try:
            # doc_type별로 임의 1개 문서 name을 쿼리로 사용 (실제로 코퍼스에 존재하므로 히트가 나와야 정상)
            for doc_type in ["qna", "action", "plan"]:
                sample_names = list(self.documents.get(sub_id, {}).get(doc_type, {}).keys())
                if not sample_names:
                    continue

                sample_q = sample_names[0]  # 첫 번째 문서명으로 테스트(원하면 random.choice로 변경)
                emb = self.bge_model.encode(
                    [sample_q],
                    return_dense=True,
                    return_sparse=True,
                    return_colbert_vecs=False,
                    max_length=8192
                )

                # sparse query 생성
                q_sparse_dict = emb["lexical_weights"][0]
                q_indices, q_values = [], []
                for idx, val in q_sparse_dict.items():
                    q_indices.append(int(idx))
                    q_values.append(float(val.item()) if hasattr(val, "item") else float(val))

                q_sparse = models.SparseVector(indices=q_indices, values=q_values)

                # dense query 생성
                q_dense = emb["dense_vecs"][0]
                if hasattr(q_dense, "tolist"):
                    q_dense = q_dense.tolist()
                elif isinstance(q_dense, np.ndarray):
                    q_dense = q_dense.astype(float).tolist()

                collection_name = collections.get(doc_type)
                if not collection_name:
                    continue

                r_sparse = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=q_sparse,
                    using="sparse",
                    limit=5,
                    with_payload=True,
                ).points

                r_dense = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=q_dense,
                    using="dense",
                    limit=5,
                    with_payload=True,
                ).points

                print(f"🧪 [SMOKE:{doc_type}] query='{sample_q}'")
                print(f"   - sparse hits={len(r_sparse)} top1={r_sparse[0].payload.get('path') if r_sparse else None}")
                print(f"   - dense  hits={len(r_dense)} top1={r_dense[0].payload.get('path') if r_dense else None}")

                # 확인 포인트:
                # - sparse hits가 0이면: (1) 컬렉션 스키마 문제 (2) query 생성 문제 (3) 업서트 문제
                # - dense는 나오는데 sparse만 0이면: sparse_vectors_config/using 이름/전송 문제를 의심
        except Exception as e:
            print(f"⚠️ 업로드 후 스모크 테스트 실패: {e}")'''

        return bge_count

    def _compute_similarity(self, query_emb: dict, doc_idx: int, doc_type: str, weights: dict = None, sub_id: int = None) -> float:
        """
        쿼리와 문서 간 유사도 계산

        Args:
            query_emb: 쿼리 임베딩 (dense, sparse, colbert)
            doc_idx: 문서 인덱스
            doc_type: 문서 타입 (qna, action, plan)
            weights: 가중치 딕셔너리

        Returns:
            하이브리드 유사도 점수
        """
        if weights is None:
            weights = self.default_weights

        # 구독키 결정
        target_sub_id = sub_id if sub_id is not None else self.sub_id
        emb_data = self.embeddings[target_sub_id][doc_type]

        # Dense similarity
        import numpy as np

        # dense_vecs가 numpy 배열인지 확인
        if isinstance(emb_data['dense_vecs'], np.ndarray):
            doc_dense = emb_data['dense_vecs'][doc_idx]
        else:
            # 리스트인 경우
            doc_dense = emb_data['dense_vecs'][doc_idx]
            if isinstance(doc_dense, list):
                doc_dense = np.array(doc_dense)

        query_dense = query_emb['dense_vecs'][0]
        if isinstance(query_dense, list):
            query_dense = np.array(query_dense)

        dense_score = float(np.dot(query_dense, doc_dense))

        # Sparse similarity
        sparse_score = self.bge_model.compute_lexical_matching_score(
            query_emb['lexical_weights'][0],
            emb_data['lexical_weights'][doc_idx]
        )

        # ColBERT similarity
        query_colbert = query_emb['colbert_vecs'][0]
        doc_colbert = emb_data['colbert_vecs'][doc_idx]

        # ColBERT 벡터 차원 조정 (1D를 2D로 변환)
        if isinstance(query_colbert, list):
            query_colbert = np.array(query_colbert, dtype=np.float32)
        if isinstance(doc_colbert, list):
            doc_colbert = np.array(doc_colbert, dtype=np.float32)

        # numpy 배열 타입 확인 및 변환
        if isinstance(query_colbert, np.ndarray):
            query_colbert = query_colbert.astype(np.float32)
        if isinstance(doc_colbert, np.ndarray):
            doc_colbert = doc_colbert.astype(np.float32)

        # 1D 벡터인 경우 2D로 reshape (평균 풀링된 경우)
        if len(query_colbert.shape) == 1:
            query_colbert = query_colbert.reshape(1, -1)
        if len(doc_colbert.shape) == 1:
            doc_colbert = doc_colbert.reshape(1, -1)

        colbert_score = self.bge_model.colbert_score(query_colbert, doc_colbert)

        # Weighted combination
        hybrid_score = (
            weights['dense'] * dense_score +
            weights['sparse'] * sparse_score +
            weights['colbert'] * colbert_score
        )

        return hybrid_score

    def strip_temporal_tokens(self, query: str) -> str:
        """검색용 쿼리에서 시간/날짜 정보를 제거하여 의미 기반 검색 정확도 향상"""
        patterns = [
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*\d{1,2}:\d{2}(:\d{2})?',  # 2025-09-23 10:00:00
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',                             # 2025-09-23
            r'\d{1,2}/\d{1,2}(/\d{2,4})?',                              # 9/23, 9/23/2025
            r'\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?',                 # 10:00 AM
            r'\d{1,2}시(\s*\d{1,2}분)?',                                 # 10시, 10시 30분
            r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{0,4}',
            r'\d{1,2}월\s*\d{1,2}일?',                                   # 9월 23일
            r'(from|to|부터|까지)\s+\d{8,14}',                           # from 20250923100000
            r'(past|within|last|최근|지난)\s+\d+\s*(hours?|days?|weeks?|months?|시간|일|주|개월)',
            r'\b\d{8,14}\b',                                              # 20250923100000
            r'\b(today|yesterday|this\s+week|this\s+month|last\s+week|last\s+month)\b',  # 상대적 시간 (영어)
            r'(오늘|어제|이번\s*주|이번\s*달|지난\s*주|지난\s*달)',        # 상대적 시간 (한국어)
        ]
        result = query
        for p in patterns:
            result = re.sub(p, ' ', result, flags=re.IGNORECASE)
        # 시간 제거 후 남은 고립된 전치사/조사 정리
        result = re.sub(r'\bon\b\s*$', ' ', result)
        result = re.sub(r'\b(from|to)\b\s+\b(from|to)\b', ' ', result, flags=re.IGNORECASE)
        result = re.sub(r'(^|\s)(부터|까지)(\s|$)', ' ', result)
        return re.sub(r'\s+', ' ', result).strip()

    def search_documents(self,
                        query: str,
                        doc_type: str = 'qna',
                        top_k: int = 5,
                        weights: dict = None,
                        sub_id: int = None,
                        sub_intent: str = None) -> List[Tuple[str, Dict, float]]:
        """
        BGE-M3 임베딩 기반 문서 검색 (Qdrant 또는 기존 방식)

        Args:
            query: 검색 쿼리
            doc_type: 문서 타입 (qna, action, plan)
            top_k: 반환할 상위 문서 수
            weights: 하이브리드 검색 가중치
            sub_id: 구독키 (None이면 self.sub_id 사용)

        Returns:
            [(파일경로, 문서데이터, 유사도점수)] 리스트
        """
        # 검색용 쿼리에서 시간 정보 제거 (임베딩 정확도 향상)
        search_query = self.strip_temporal_tokens(query)
        if search_query != query and getattr(self, "debug_qdrant_hybrid", False):
            print(f"🕐 시간 제거: '{query}' → '{search_query}'")

        # Qdrant 메모리 사용 가능한 경우 우선 사용
        if self.use_qdrant_memory and self.qdrant_client:
            return self._search_documents_qdrant(search_query, doc_type, top_k, weights, sub_id, sub_intent)
        else:
            return self._search_documents_legacy(search_query, doc_type, top_k, weights, sub_id)

    def _search_documents_qdrant(
        self,
        query: str,
        doc_type: str = "qna",
        top_k: int = 5,
        weights: dict = None,
        sub_id: int = None,
        sub_intent: str = None,
    ) -> List[Tuple[str, Dict, float]]:
        """
        Qdrant Query API를 사용한 3-stage 검색:
        (1) dense + sparse prefetch
        (2) RRF fusion으로 후보군 확정
        (3) ColBERT(multivector)로 후보군만 rerank

        [단일 컬렉션 + sub_id/doc_type 필터 방식]
        - 컬렉션: bge (단일)
        - 검색 시 필터: (sub_id=1 OR sub_id=target) AND doc_type={qna|action|plan}

        [DEBUG 사용법]
        - self.debug_qdrant_hybrid = True  로 설정하면,
            dense-only / sparse-only / hybrid-RRF / (최종)colbert 결과를 비교 출력합니다.
        - 운영에서는 False로 두세요(로그 과다/성능 영향).
        """

        target_sub_id = sub_id if sub_id is not None else self.sub_id
        # 단일 컬렉션 사용
        collection_name = qdrant_collection()
        # ====== DEBUG FLAG (없으면 False) ======
        debug = bool(getattr(self, "debug_qdrant_hybrid", False))

        if debug:
            print(f"🔍 Qdrant에서 BGE 문서 검색: 구독 {target_sub_id}, 타입 {doc_type}, 컬렉션 {collection_name}, 질문 : {query}")

        try:
            import numpy as np
            import time as _time
            from qdrant_client import models

            # -----------------------------
            # 필터 생성 (sub_id + doc_type)
            # -----------------------------
            # sub_id 필터: 공통=1 또는 해당 구독
            sub_id_conditions = [
                models.FieldCondition(key="sub_id", match=models.MatchValue(value=1)),  # 공통 데이터
            ]
            if target_sub_id != 1:
                sub_id_conditions.append(
                    models.FieldCondition(key="sub_id", match=models.MatchValue(value=target_sub_id))  # 개별 데이터
                )

            # doc_type 필터 (단일 문자열 또는 리스트 지원)
            if isinstance(doc_type, list):
                doc_type_filter = models.Filter(
                    should=[
                        models.FieldCondition(key="doc_type", match=models.MatchValue(value=dt))
                        for dt in doc_type
                    ]
                )
            else:
                doc_type_filter = models.FieldCondition(key="doc_type", match=models.MatchValue(value=doc_type))

            # enabled 필터 (비활성화된 프롬프트 제외)
            enabled_filter = models.FieldCondition(key="enabled", match=models.MatchValue(value=True))

            # 최종 필터: sub_id(OR) AND doc_type AND enabled
            base_sub_id_filter = models.Filter(
                must=[
                    models.Filter(should=sub_id_conditions),
                    doc_type_filter,
                    enabled_filter
                ]
            )

            # -----------------------------
            # 0) 임베딩 생성 (쿼리 길이 제한: ColBERT 벡터 크기 → Qdrant 페이로드 제한 방지)
            # -----------------------------
            MAX_QUERY_LENGTH = 1000
            search_query_text = query
            if len(search_query_text) > MAX_QUERY_LENGTH:
                print(f"⚠️ 검색 쿼리 길이 제한: {len(search_query_text)}자 → {MAX_QUERY_LENGTH}자로 truncate")
                search_query_text = search_query_text[:MAX_QUERY_LENGTH]

            _emb_start = time.time()
            query_emb = self.bge_model.encode(
                [search_query_text],
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
                max_length=512,
            )
            self._last_embed_time = time.time() - _emb_start
            if debug:
                print(f"⏱️ BGE-M3 쿼리 임베딩 완료 {self._last_embed_time:.3f}초 (Qdrant 검색용)")

            # Dense
            dense_query = query_emb["dense_vecs"][0]
            if hasattr(dense_query, "tolist"):
                dense_query = dense_query.tolist()
            elif isinstance(dense_query, np.ndarray):
                dense_query = dense_query.astype(float).tolist()

            # Sparse (lexical_weights: dict[token_id -> weight])
            sparse_query_dict = query_emb["lexical_weights"][0]
            sparse_indices, sparse_values = [], []
            for idx, val in sparse_query_dict.items():
                sparse_indices.append(int(idx))
                sparse_values.append(float(val.item()) if hasattr(val, "item") else float(val))
            sparse_query = models.SparseVector(indices=sparse_indices, values=sparse_values)

            # ColBERT (multivector: List[List[float]])
            colbert_query = query_emb["colbert_vecs"][0]
            if hasattr(colbert_query, "tolist"):
                colbert_query_matrix = colbert_query.tolist()
            elif isinstance(colbert_query, np.ndarray):
                colbert_query_matrix = colbert_query.astype(float).tolist()
            else:
                colbert_query_matrix = list(colbert_query)

            # 1D면 2D로 보정
            if (
                isinstance(colbert_query_matrix, list)
                and len(colbert_query_matrix) > 0
                and not isinstance(colbert_query_matrix[0], list)
            ):
                colbert_query_matrix = [colbert_query_matrix]
            
            # ---------------------------------------------------------
            # ROUTER 1) 보안 ID 질의면: filter 기반 exact retrieval 우선
            # ---------------------------------------------------------
            ids_in_query = parse_security_ids(query)
            id_filter = build_id_filter(ids_in_query)

            if id_filter is not None:
                # ID 질의는 의미검색보다 "정확히" 맞는 문서를 먼저 확보하는게 핵심
                # sub_id 필터와 ID 필터 결합
                combined_id_filter = models.Filter(must=[base_sub_id_filter, id_filter])
                _qdrant_start = _time.perf_counter()

                # 1) filter + dense (정렬)
                id_first = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=dense_query,
                    using="dense",
                    query_filter=combined_id_filter,
                    limit=max(top_k, 10),
                    with_payload=True,
                )
                id_pts = getattr(id_first, "points", id_first)

                # 2) filter + sparse (보강) - sparse가 비어도 상관 없음
                id_second = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=sparse_query,
                    using="sparse",
                    query_filter=combined_id_filter,
                    limit=max(top_k, 10),
                    with_payload=True,
                )
                id_pts2 = getattr(id_second, "points", id_second)

                # 3) 합치고(중복 제거) 점수 기준으로 정렬
                merged = {}
                for p in list(id_pts) + list(id_pts2):
                    fp = (p.payload or {}).get("path", "")
                    if not fp:
                        continue
                    # 동일 문서가 여러 경로로 들어오면 최고 점수 유지
                    if fp not in merged or float(p.score) > float(merged[fp].score):
                        merged[fp] = p

                cand = list(merged.values())
                cand.sort(key=lambda x: float(x.score), reverse=True)

                # 4) ID 질의는 여기서 바로 반환해도 되고,
                #    문서가 많을 때만 colbert로 "같은 ID 문서들" 사이 rerank를 걸어도 됨
                if len(cand) > top_k:
                    # filter + colbert rerank (후보가 있을 때만)
                    # Qdrant query_points는 후보 제한을 직접 넣기 어렵기 때문에,
                    # ID filter를 그대로 두고 colbert로 다시 질의
                    id_rerank = self.qdrant_client.query_points(
                        collection_name=collection_name,
                        query=colbert_query_matrix,
                        using="colbert",
                        query_filter=combined_id_filter,
                        with_payload=True,
                        limit=top_k,
                    )
                    points = getattr(id_rerank, "points", id_rerank)
                else:
                    points = cand[:top_k]

                if debug:
                    print("\n=== router: ID-filter path ===")
                    print(f"[DEBUG ids] {ids_in_query.get('norm_ids')}")
                    for i, p in enumerate(points[:10], 1):
                        payload = p.payload or {}
                        print(f"{i:02d}. score={float(p.score):.6f}  file_path={payload.get('path')}")

                # 결과 포맷팅 후 즉시 반환
                results: List[Tuple[str, Dict, float]] = []
                for p in points:
                    payload = p.payload or {}
                    file_path = payload.get("path", "")
                    doc_data = {
                        "content": payload.get("content", ""),
                        "file_path": file_path,
                        "doc_type": doc_type,
                        "description": payload.get("description", ""),
                        "sub_id": target_sub_id,
                    }
                    results.append((file_path, doc_data, float(p.score)))

                self._last_qdrant_time = _time.perf_counter() - _qdrant_start
                if debug:
                    print(f"✅ Qdrant 검색 완료(ID filter): {len(results)}개 문서 반환")
                return results

            # ---------------------------------------------------------
            # ROUTER 1.5) 명령어/함수 문서 검색 (LLM이 COMMAND_REF로 판정한 경우에만)
            # ---------------------------------------------------------
            if debug:
                print(f"[DEBUG] sub_intent 값 확인: '{sub_intent}'")
            cmd_tokens = []
            cmd_filter = None

            if sub_intent == 'COMMAND_REF':
                cmd_tokens = self.extract_command_like_tokens_strict(query)
                cmd_filter = build_command_filter(cmd_tokens)
                if debug:
                    print(f"[DEBUG] COMMAND_REF 모드: cmd_tokens={cmd_tokens} cmd_filter={cmd_filter}")
            else:
                if debug:
                    print(f"[DEBUG] GENERAL 모드: 명령어 필터 검색 건너뜀")

            if cmd_filter is not None:
                # sub_id 필터와 cmd 필터 결합
                combined_cmd_filter = models.Filter(must=[base_sub_id_filter, cmd_filter])
                _qdrant_start = _time.perf_counter()

                # 1) filter + sparse (정확 키워드 성격에 유리)
                r_sparse = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=sparse_query,
                    using="sparse",
                    query_filter=combined_cmd_filter,
                    limit=max(top_k, 20),
                    with_payload=True,
                )
                pts_sparse = getattr(r_sparse, "points", r_sparse)

                # 2) filter + dense (보강)
                r_dense = self.qdrant_client.query_points(
                    collection_name=collection_name,
                    query=dense_query,
                    using="dense",
                    query_filter=combined_cmd_filter,
                    limit=max(top_k, 20),
                    with_payload=True,
                )
                pts_dense = getattr(r_dense, "points", r_dense)

                # 3) merge(중복 제거) 후 점수 내림차순
                merged = {}
                for p in list(pts_sparse) + list(pts_dense):
                    fp = (p.payload or {}).get("path", "")
                    if not fp:
                        continue
                    if fp not in merged or float(p.score) > float(merged[fp].score):
                        merged[fp] = p

                cand = list(merged.values())
                cand.sort(key=lambda x: float(x.score), reverse=True)

                # 4) 후보가 충분하면 colbert로 같은 후보군 내 정렬(의미 확장 최소화 목적)
                #    - 여기서는 filter를 유지하고 colbert로 top_k만 뽑는다.
                if len(cand) > top_k:
                    r_col = self.qdrant_client.query_points(
                        collection_name=collection_name,
                        query=colbert_query_matrix,
                        using="colbert",
                        query_filter=combined_cmd_filter,
                        with_payload=True,
                        limit=top_k,
                    )
                    points = getattr(r_col, "points", r_col)
                else:
                    points = cand[:top_k]

                if debug:
                    print("\n=== router: COMMAND/FUNCTION filter path ===")
                    print(f"[DEBUG cmd_tokens] {cmd_tokens}")
                    for i, p in enumerate(points[:10], 1):
                        payload = p.payload or {}
                        print(f"{i:02d}. score={float(p.score):.6f}  file_path={payload.get('path')}  "
                            f"cmd={payload.get('command_name') or payload.get('func_name')}")

                results: List[Tuple[str, Dict, float]] = []
                for p in points:
                    payload = p.payload or {}
                    file_path = payload.get("path", "")
                    doc_data = {
                        "content": payload.get("content", ""),
                        "file_path": file_path,
                        "doc_type": doc_type,
                        "description": payload.get("description", ""),
                        "sub_id": target_sub_id,
                        "mode_brand": payload.get("mode_brand"),
                        "command_name": payload.get("command_name"),
                        "func_name": payload.get("func_name"),
                    }
                    results.append((file_path, doc_data, float(p.score)))

                self._last_qdrant_time = _time.perf_counter() - _qdrant_start
                if debug:
                    print(f"✅ Qdrant 검색 완료(COMMAND/FUNCTION filter): {len(results)}개 문서 반환")
                return results

            # -----------------------------
            # 1) 후보군 예산 (3844개 규모 최적화)
            # -----------------------------
            B = max(80, min(150, top_k * 10))  # top_k=10 -> 100
            dense_limit = int(B * 0.45)   # 45
            sparse_limit = int(B * 0.45)  # 45
            colbert_limit = int(B * 0.10) # 10

            # =========================================================
            # DEBUG ROUTINE: dense / sparse / colbert / 3-way RRF 비교
            # =========================================================
            if debug:
                def _top_print(title: str, pts, n: int = 10):
                    print(f"\n=== {title} (top {min(n, len(pts))}/{len(pts)}) ===")
                    for i, p in enumerate(pts[:n], 1):
                        payload = getattr(p, "payload", None) or {}
                        print(f"{i:02d}. score={float(p.score):.6f}  file_path={payload.get('path')}")

                def _to_fp_set(pts):
                    out = set()
                    for p in pts:
                        payload = getattr(p, "payload", None) or {}
                        out.add(payload.get("path"))
                    return out

                _dbg_limit = min(50, max(20, top_k * 5))

                # dense-only
                d_res = self.qdrant_client.query_points(
                    collection_name=collection_name, query=dense_query, using="dense",
                    query_filter=base_sub_id_filter, limit=_dbg_limit, with_payload=True)
                d_pts = getattr(d_res, "points", d_res)

                # sparse-only
                s_res = self.qdrant_client.query_points(
                    collection_name=collection_name, query=sparse_query, using="sparse",
                    query_filter=base_sub_id_filter, limit=_dbg_limit, with_payload=True)
                s_pts = getattr(s_res, "points", s_res)

                # colbert-only
                c_res = self.qdrant_client.query_points(
                    collection_name=collection_name, query=colbert_query_matrix, using="colbert",
                    query_filter=base_sub_id_filter, limit=_dbg_limit, with_payload=True)
                c_pts = getattr(c_res, "points", c_res)

                # 출력
                print("\n🧪 [DEBUG] Query sparse stats:"
                    f" nz={len(sparse_query.indices)}"
                    f" max={max(sparse_query.values) if sparse_query.values else 0.0:.6f}")

                _top_print("dense-only", d_pts, n=10)
                _top_print("sparse-only", s_pts, n=10)
                _top_print("colbert-only", c_pts, n=10)

                # 교집합/차집합 요약
                D, S, C = _to_fp_set(d_pts), _to_fp_set(s_pts), _to_fp_set(c_pts)
                print(
                    f"\n[DEBUG sets]"
                    f" |D|={len(D)} |S|={len(S)} |C|={len(C)}"
                    f" |D∩S|={len(D & S)} |D∩C|={len(D & C)} |S∩C|={len(S & C)}"
                    f" |D∩S∩C|={len(D & S & C)}"
                )

            # -----------------------------
            # 3) 3-way RRF (dense + sparse + ColBERT)
            # -----------------------------
            # 개별 벡터 검색 소요시간 측정
            if debug:
                _td0 = _time.perf_counter()
                self.qdrant_client.query_points(
                    collection_name=collection_name, query=dense_query, using="dense",
                    query_filter=base_sub_id_filter, limit=dense_limit, with_payload=False)
                _td1 = _time.perf_counter()

                _ts0 = _time.perf_counter()
                self.qdrant_client.query_points(
                    collection_name=collection_name, query=sparse_query, using="sparse",
                    query_filter=base_sub_id_filter, limit=sparse_limit, with_payload=False)
                _ts1 = _time.perf_counter()

                _tc0 = _time.perf_counter()
                self.qdrant_client.query_points(
                    collection_name=collection_name, query=colbert_query_matrix, using="colbert",
                    query_filter=base_sub_id_filter, limit=colbert_limit, with_payload=False)
                _tc1 = _time.perf_counter()

                print(f"⏱️ [개별 벡터] dense={(_td1-_td0)*1000:.1f}ms, sparse={(_ts1-_ts0)*1000:.1f}ms, colbert={(_tc1-_tc0)*1000:.1f}ms")

            _t0 = _time.perf_counter()
            res = self.qdrant_client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(query=dense_query, using="dense", limit=dense_limit, filter=base_sub_id_filter),
                    models.Prefetch(query=sparse_query, using="sparse", limit=sparse_limit, filter=base_sub_id_filter),
                    models.Prefetch(query=colbert_query_matrix, using="colbert", limit=colbert_limit, filter=base_sub_id_filter),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                with_payload=True,
                limit=top_k,
            )
            _t1 = _time.perf_counter()
            self._last_qdrant_time = _t1 - _t0
            if debug:
                print(f"⏱️ [3-way RRF] prefetch+fusion: {self._last_qdrant_time*1000:.1f}ms (dense={dense_limit}, sparse={sparse_limit}, colbert={colbert_limit}, top_k={top_k})")
            points = getattr(res, "points", res)

            # DEBUG: 최종 결과 출력
            if debug:
                print(f"\n=== final (3-way RRF: dense+sparse+colbert) ===")
                for i, p in enumerate(points[: min(10, len(points))], 1):
                    payload = p.payload or {}
                    print(f"{i:02d}. score={float(p.score):.6f}  file_path={payload.get('path')}")

            # 결과 포맷팅
            results: List[Tuple[str, Dict, float]] = []

            for p in points:
                payload = p.payload or {}
                file_path = payload.get("path", "")
                score = float(p.score)

                doc_data = {
                    "content": payload.get("content", ""),
                    "file_path": file_path,
                    "doc_type": doc_type,
                    "description": payload.get("description", ""),
                    "sub_id": target_sub_id,
                }

                results.append((file_path, doc_data, score))
            if debug:
                print(f"✅ Qdrant 검색 완료: {len(results)}개 문서 반환 (B={B}, dense={dense_limit}, sparse={sparse_limit}, colbert={colbert_limit})")
            return results

        except Exception as e:
            print(f"❌ Qdrant 검색 실패: {e}, 기존 방식으로 폴백")
            return self._search_documents_legacy(query, doc_type, top_k, weights, sub_id)

    def _search_documents_legacy(self,
                                query: str,
                                doc_type: str = 'qna',
                                top_k: int = 5,
                                weights: dict = None,
                                sub_id: int = None) -> List[Tuple[str, Dict, float]]:
        """
        기존 numpy 방식의 BGE 문서 검색 (백업용)
        """
        # 구독키 결정
        target_sub_id = sub_id if sub_id is not None else self.sub_id

        # 구독키별 임베딩 확인
        if target_sub_id not in self.embeddings or doc_type not in self.embeddings[target_sub_id]:
            print(f"⚠️ No embeddings loaded for sub_id: {target_sub_id}, doc_type: {doc_type}")
            return []

        # 쿼리 임베딩 생성
        _emb_start = time.time()
        query_emb = self.bge_model.encode(
            [query],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            max_length=8192
        )
        if getattr(self, "debug_qdrant_hybrid", False):
            print(f"⏱️ BGE-M3 쿼리 임베딩 완료 {time.time() - _emb_start:.3f}초 (로컬 검색용)")

        # 각 문서와 유사도 계산
        doc_scores = []
        emb_data = self.embeddings[target_sub_id][doc_type]
        num_docs = len(emb_data.get('dense_vecs', []))

        for i in range(num_docs):
            # 유사도 계산
            similarity = self._compute_similarity(query_emb, i, doc_type, weights, target_sub_id)

            # 문서 정보 추출
            if 'documents' in emb_data and i < len(emb_data['documents']):
                doc_info = emb_data['documents'][i]
                original_file_path = doc_info.get('file_path', f'doc_{i}')

                # 문서 타입 prefix가 없으면 추가 (QNA 모드 필터링을 위해)
                if not original_file_path.startswith(f'{doc_type}/'):
                    file_path = f'{doc_type}/{original_file_path}'
                else:
                    file_path = original_file_path

                # 문서 데이터 구성
                doc_data = {
                    'content': doc_info.get('content', ''),
                    'file_path': file_path,
                    'doc_type': doc_type,
                    'description': doc_info.get('description', ''),
                    'sub_id': target_sub_id
                }
            else:
                file_path = f'{doc_type}/doc_{i}'
                doc_data = {
                    'content': '',
                    'file_path': file_path,
                    'doc_type': doc_type,
                    'metadata': {}
                }

            doc_scores.append((file_path, doc_data, similarity))

        # 유사도 기준 정렬
        doc_scores.sort(key=lambda x: x[2], reverse=True)

        return doc_scores[:top_k]

    def search_with_faiss_qna(self,
                             query: str,
                             top_k: int = 5,
                             sub_id: int = None,
                             enable_synonym: bool = True) -> List[Tuple[str, dict, float]]:
        """
        기존 aibot_rag_module과의 호환성을 위한 메서드
        QnA 문서 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 문서 수
            sub_id: 구독 ID (현재 미사용)
            enable_synonym: 동의어 확장 여부 (현재 미사용)

        Returns:
            [(파일경로, 문서데이터, 유사도점수)] 리스트
        """
        return self.search_documents(query, doc_type=['qna', 'action', 'plan'], top_k=top_k, sub_id=sub_id)

    def search_hybrid(self,
                     query: str,
                     top_k: int = 5,
                     doc_types: List[str] = None,
                     sub_id: int = None) -> List[Tuple[str, Dict, float]]:
        """
        여러 문서 타입에서 하이브리드 검색

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 문서 수
            doc_types: 검색할 문서 타입 리스트
            sub_id: 구독키 (None이면 self.sub_id 사용)

        Returns:
            [(파일경로, 문서데이터, 유사도점수)] 리스트
        """
        if doc_types is None:
            doc_types = ['qna', 'action', 'plan']

        all_results = []

        # 각 문서 타입별로 검색
        for doc_type in doc_types:
            # Qdrant 사용 시 embeddings 사전 체크 불필요 (Qdrant 내부에서 필터링)
            if self.use_qdrant_memory and self.qdrant_client:
                results = self.search_documents(query, doc_type=doc_type, top_k=top_k, sub_id=sub_id, sub_intent=None)
                all_results.extend(results)
            elif doc_type in self.embeddings and self.embeddings[doc_type] is not None:
                results = self.search_documents(query, doc_type=doc_type, top_k=top_k, sub_id=sub_id, sub_intent=None)
                all_results.extend(results)

        # 유사도 기준 정렬
        all_results.sort(key=lambda x: x[2], reverse=True)

        return all_results[:top_k]

    def read_file_content(self, file_key: str, sub_id: int = None) -> str:
        """
        파일 키로 문서 내용 읽기 (Qdrant에서 조회)

        Args:
            file_key: 문서 파일 키 (예: qna/xxx.yaml)
            sub_id: 구독 ID

        Returns:
            문서 내용 문자열
        """
        if sub_id is None:
            sub_id = self.sub_id

        # Qdrant에서 file_key로 문서 검색
        if self.qdrant_client:
            try:
                from qdrant_client import models

                # file_key 또는 path로 검색
                result = self.qdrant_client.scroll(
                    collection_name=qdrant_collection(),
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="sub_id",
                                match=models.MatchValue(value=sub_id)
                            ),
                            models.Filter(
                                should=[
                                    models.FieldCondition(
                                        key="file_key",
                                        match=models.MatchValue(value=file_key)
                                    ),
                                    models.FieldCondition(
                                        key="path",
                                        match=models.MatchValue(value=file_key)
                                    ),
                                ]
                            )
                        ]
                    ),
                    with_payload=True,
                    with_vectors=False,
                    limit=1
                )

                if result[0]:
                    payload = result[0][0].payload
                    content = payload.get("content") or payload.get("text", "")
                    if content:
                        return content

            except Exception as e:
                print(f"⚠️ Qdrant에서 문서 조회 실패 ({file_key}): {e}")

        # 메모리 캐시에서 검색
        for dtype in ['qna', 'action', 'plan']:
            if file_key in self.documents.get(dtype, {}):
                doc = self.documents[dtype][file_key]
                return doc.get("content", "") if isinstance(doc, dict) else str(doc)

        return f"[문서를 찾을 수 없습니다: {file_key}]"

    def get_document_content(self, file_path: str, doc_type: str = None) -> Optional[Dict]:
        """
        파일 경로로 문서 내용 가져오기

        Args:
            file_path: 문서 파일 경로
            doc_type: 문서 타입 (None이면 모든 타입에서 검색)

        Returns:
            문서 데이터 딕셔너리 또는 None
        """
        if doc_type:
            return self.documents.get(doc_type, {}).get(file_path)

        # 모든 타입에서 검색
        for dtype in ['qna', 'action', 'plan']:
            if file_path in self.documents.get(dtype, {}):
                return self.documents[dtype][file_path]

        return None

    def update_weights(self, dense: float = None, sparse: float = None, colbert: float = None):
        """
        하이브리드 검색 가중치 업데이트

        Args:
            dense: Dense 벡터 가중치
            sparse: Sparse 벡터 가중치
            colbert: ColBERT 가중치
        """
        if dense is not None:
            self.default_weights['dense'] = dense
        if sparse is not None:
            self.default_weights['sparse'] = sparse
        if colbert is not None:
            self.default_weights['colbert'] = colbert

        # 정규화 (합이 1이 되도록)
        total = sum(self.default_weights.values())
        if total > 0:
            for key in self.default_weights:
                self.default_weights[key] /= total

        print(f"🔧 Updated weights: {self.default_weights}")

    def find_similar_docs(self,
                         query: str,
                         top_k: int = 5,
                         max_token_count: int = 6000,
                         intent: str = None,
                         sub_intent: str = None,
                         sub_id: int = None,
                         doc_type: str = None) -> List[Tuple[str, Dict, float]]:
        """
        기존 RAGSystem과의 완벽한 호환성을 위한 메서드

        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 문서 수
            max_token_count: 최대 토큰 수 (현재 미사용, 호환성용)
            intent: 의도 타입 (qna, action, plan 등)
            sub_intent: 하위 의도
            sub_id: 구독 ID (현재 미사용, 호환성용)
            doc_type: 문서 타입

        Returns:
            [(파일경로, 문서딕셔너리, 유사도점수)] 리스트
        """
        # intent에 따라 검색 범위 결정 (Qdrant 1회 호출)
        # QNA: 전체 검색 (qna, action, plan)
        # ACTION/PLAN: 해당 타입만 검색
        if intent and 'action' in intent.lower():
            search_doc_type = 'action'
        elif intent and 'plan' in intent.lower():
            search_doc_type = 'plan'
        else:
            search_doc_type = ['qna', 'action']

        results = self.search_documents(query, doc_type=search_doc_type, top_k=top_k, sub_id=sub_id, sub_intent=sub_intent)

        # 반환 형식을 기존 RAGSystem과 동일하게 변환
        # (file_path, content_dict, score) 형태로 변환
        _file_read_start = time.time()
        formatted_results = []
        for file_path, doc_data, score in results:
            # 실제 YAML 파일을 읽어서 파싱된 형태로 반환
            try:
                base_path = "/data/ai-agent/docs/aibot/yaml/"
                full_path = os.path.join(base_path, file_path)

                # YAML 파일 읽고 문자열로 반환 (기존 모듈과 동일한 인터페이스)
                if os.path.exists(full_path):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        yaml_content_str = f.read()  # YAML을 문자열로 읽음
                    formatted_results.append((file_path, yaml_content_str, score))
                else:
                    # 파일이 없는 경우 임베딩된 텍스트 사용
                    if isinstance(doc_data, dict):
                        content = doc_data.get('content', '')
                    else:
                        content = str(doc_data)
                    formatted_results.append((file_path, content, score))

            except Exception as e:
                print(f"⚠️ Error loading {file_path}: {e}")
                # 오류 시 임베딩된 텍스트 사용
                if isinstance(doc_data, dict):
                    content = doc_data.get('content', '')
                else:
                    content = str(doc_data)

                content_dict = {
                    'content': content,
                    'file_content_loaded': True
                }
                formatted_results.append((file_path, content_dict, score))

        self._last_file_read_time = time.time() - _file_read_start

        # 로깅 (디버깅용)
        if len(formatted_results) > 0 and getattr(self, "debug_qdrant_hybrid", False):
            print(f"🔍 [BGE-M3] Found {len(formatted_results)} documents for query: '{query[:50]}...'")
            for i, (path, _, score) in enumerate(formatted_results[:3], 1):
                print(f"   {i}. [{score:.4f}] {path}")

        return formatted_results

    def find_similar_docs_test(self,
                               query: str,
                               top_k: int = 5,
                               max_token_count: int = 6000,
                               base_token_count: int = 1000,
                               intent: str = None,
                               sub_intent: str = None,
                               temporary_embeddings: Dict[str, Dict] = None,
                               sub_id: int = None) -> List[Tuple[str, str, float]]:
        """
        임시 임베딩으로 증분 테스트 (Qdrant 임시 포인트 방식)

        temporary_embeddings 형식:
          - BGE 모드 (bge_mode=True): {file_key: {content, test_id, ...}}
            → 이 메서드에서 직접 BGE 임베딩 생성
          - FAISS 호환: {file_key: {vector: {...}, ...}}
            → 이미 생성된 벡터 사용

        흐름:
          1. 임시 데이터 임베딩 생성 (BGE-M3)
          2. Qdrant에 임시 포인트 upsert
          3. 기존 + 임시 데이터 합쳐서 검색
          4. 임시 포인트 삭제 (원상복구)
        """
        import hashlib
        import yaml
        from qdrant_client import models as qd_models

        print(f"🧪 [BGE 증분 테스트] 검색 실행")
        print(f"   - 쿼리: '{query}'")
        print(f"   - 임시 데이터: {len(temporary_embeddings) if temporary_embeddings else 0}개")
        print(f"   - sub_id: {sub_id}")

        temp_point_ids = []

        try:
            # 1. 임시 데이터 임베딩 생성 + Qdrant 삽입
            if temporary_embeddings:
                print(f"🔄 [BGE] 임시 데이터 임베딩 생성 및 Qdrant 삽입 중...")
                target_sub_id = sub_id if sub_id is not None else self.sub_id

                for file_key, temp_data in temporary_embeddings.items():
                    try:
                        # --- 임베딩 생성 ---
                        is_bge_precomputed = temp_data.get('bge_mode', False) and isinstance(temp_data.get('vector', None), dict)

                        if isinstance(temp_data.get('vector', None), dict) and 'dense' in temp_data['vector']:
                            # 이미 생성된 BGE 벡터 사용
                            packed_vector = temp_data['vector']
                        else:
                            # 원본 content로부터 BGE-M3 임베딩 직접 생성
                            content = temp_data.get('content', '')
                            if not content:
                                print(f"   ⚠️ content 없음: {file_key}")
                                continue

                            embedding_text = self._extract_embedding_text_for_test(content)
                            emb = self.bge_model.encode(
                                [embedding_text],
                                return_dense=True,
                                return_sparse=True,
                                return_colbert_vecs=True,
                                max_length=192,  # 임시 테스트용 — ColBERT 토큰 절감
                            )

                            packed_vector = {
                                'dense': emb['dense_vecs'][0].tolist() if hasattr(emb['dense_vecs'][0], 'tolist') else list(emb['dense_vecs'][0]),
                                'sparse': dict(zip(
                                    emb['lexical_weights'][0].keys() if isinstance(emb['lexical_weights'][0], dict) else [],
                                    emb['lexical_weights'][0].values() if isinstance(emb['lexical_weights'][0], dict) else [],
                                )) if emb.get('lexical_weights') else {},
                                'colbert': emb['colbert_vecs'][0].tolist() if emb.get('colbert_vecs') is not None and hasattr(emb['colbert_vecs'][0], 'tolist') else (list(emb['colbert_vecs'][0]) if emb.get('colbert_vecs') is not None else []),
                            }
                            print(f"   🔧 임베딩 생성: {file_key}")

                        # --- 벡터 변환 ---
                        dense = packed_vector.get('dense', [])
                        if hasattr(dense, 'tolist'):
                            dense = dense.tolist()
                        if not dense or len(dense) != 1024:
                            print(f"   ⚠️ 잘못된 dense 차원: {file_key}")
                            continue

                        # point ID (임시용, 충돌 방지)
                        temp_guid = f"temp_{file_key}_{temp_data.get('test_id', '')}"
                        point_id = int(hashlib.sha256(temp_guid.encode()).hexdigest()[:15], 16)
                        temp_point_ids.append(point_id)

                        named_vectors = {"dense": dense}

                        # colbert (임시 테스트이므로 192토큰으로 제한 — 10MB payload 초과 방지)
                        MAX_COLBERT_TOKENS_TEST = 192
                        colbert = packed_vector.get('colbert', [])
                        if hasattr(colbert, 'tolist'):
                            colbert = colbert.tolist()
                        if colbert:
                            if not isinstance(colbert[0], list):
                                colbert = [colbert]
                            if len(colbert) > MAX_COLBERT_TOKENS_TEST:
                                colbert = colbert[:MAX_COLBERT_TOKENS_TEST]
                            named_vectors["colbert"] = colbert

                        # sparse
                        sparse = packed_vector.get('sparse', {})
                        if sparse:
                            s_indices = [int(k) for k in sparse.keys()]
                            s_values = [float(v.item()) if hasattr(v, 'item') else float(v) for v in sparse.values()]
                            if s_indices:
                                named_vectors["sparse"] = qd_models.SparseVector(
                                    indices=s_indices, values=s_values
                                )

                        # doc_type / payload
                        doc_type = file_key.split("/")[0] if "/" in file_key else "qna"
                        content_str = temp_data.get('content', '')

                        payload = {
                            "sub_id": target_sub_id,
                            "doc_type": doc_type,
                            "file_key": file_key,
                            "guid": temp_guid,
                            "enabled": True,
                            "name": temp_data.get("filename", file_key.split("/")[-1] if "/" in file_key else file_key),
                            "type": doc_type,
                            "filename": temp_data.get("filename", ""),
                            "path": file_key,
                            "description": temp_data.get("question", ""),
                            "content": content_str,
                            "text": content_str,
                            "temp_data": True,
                        }

                        self.qdrant_client.upsert(
                            collection_name=qdrant_collection(),
                            points=[qd_models.PointStruct(
                                id=point_id,
                                vector=named_vectors,
                                payload=payload,
                            )]
                        )
                        print(f"   ✅ 임시 삽입: {file_key} (id={point_id})")

                    except Exception as e:
                        print(f"   ❌ 임시 삽입 실패 ({file_key}): {e}")

            # 2. 검색 (기존 + 임시 데이터 함께)
            print(f"\n🔍 [BGE] 확장된 데이터에서 검색 수행")
            results = self.find_similar_docs(
                query=query,
                top_k=top_k,
                max_token_count=max_token_count,
                intent=intent,
                sub_intent=sub_intent,
                sub_id=sub_id,
            )

            # 검색 결과 분석
            if results and temporary_embeddings:
                print(f"\n📋 검색 결과 분석:")
                print(f"   - 총 검색 결과: {len(results)}개")
                temp_count = sum(1 for fk, _, _ in results if fk in temporary_embeddings)
                print(f"   - 임시 데이터 검색 결과: {temp_count}/{len(results)}개")

            return results

        finally:
            # 3. 임시 포인트 삭제 (원상복구)
            if temp_point_ids:
                try:
                    self.qdrant_client.delete(
                        collection_name=qdrant_collection(),
                        points_selector=qd_models.PointIdsList(points=temp_point_ids),
                    )
                    print(f"\n🔙 [BGE] 임시 포인트 {len(temp_point_ids)}개 삭제 완료")
                except Exception as e:
                    print(f"⚠️ 임시 포인트 삭제 실패: {e}")

            print(f"✅ [BGE] 증분 테스트 완료")

    def _extract_embedding_text_for_test(self, content: str) -> str:
        """YAML content에서 임베딩용 텍스트 추출"""
        import yaml
        try:
            yaml_data = yaml.safe_load(content)
            if not isinstance(yaml_data, dict):
                return content

            parts = []
            for field, prefix in [
                ('question', '질문'),
                ('cot', '추론과정'),
                ('spec', '명세'),
                ('answer', '답변'),
                ('description', '설명'),
            ]:
                value = yaml_data.get(field, '')
                if value:
                    if isinstance(value, list):
                        value = ' '.join(str(item) for item in value)
                    parts.append(f"{prefix}: {str(value).strip()}")

            return ' '.join(parts) if parts else content
        except Exception:
            return content

    def get_stats(self) -> Dict:
        """
        시스템 통계 반환

        Returns:
            통계 정보 딕셔너리
        """
        stats = {
            'model': 'BGE-M3',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'weights': self.default_weights.copy(),
            'embeddings': {}
        }

        for doc_type in ['qna', 'action', 'plan']:
            if self.embeddings[doc_type] is not None:
                stats['embeddings'][doc_type] = {
                    'loaded': True,
                    'count': len(self.embeddings[doc_type].get('dense_vecs', []))
                }
            else:
                stats['embeddings'][doc_type] = {
                    'loaded': False,
                    'count': 0
                }

        return stats

    def _init_bge_vector_db(self):
        """BGE 벡터 DB 초기화 (사용하지 않음)"""
        # BGE 벡터 DB를 사용하지 않으므로 비활성화
        self.bge_vector_manager = None
        self.use_bge_vector_db = False
        print("ℹ️ BGE 벡터 DB 비활성화 - 메모리 기반 BGE 검색 사용")


# 테스트 코드
if __name__ == "__main__":
    # BGE-M3 RAG 시스템 초기화
    rag_system = RAGSystemBGE()

    # 시스템 통계 출력
    stats = rag_system.get_stats()
    print("\n📊 System Stats:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    # 테스트 쿼리
    test_queries = [
        "로그 검색 쿼리 문법 사용법",
        "SQL Injection 공격 방법",
        "최근 7일간 트래픽 추이 보기"
    ]

    print("\n" + "="*80)
    print("🧪 Testing BGE-M3 RAG System")
    print("="*80)

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-"*60)

        # QnA 검색
        results = rag_system.search_with_faiss_qna(query, top_k=3)

        if results:
            print(f"📄 Top {len(results)} results:")
            for i, (file_path, doc_data, score) in enumerate(results, 1):
                print(f"  {i}. [{score:.4f}] {file_path}")
                content_preview = doc_data.get('content', '')[:100]
                if content_preview:
                    print(f"     {content_preview}...")
        else:
            print("  No results found")

    # 하이브리드 검색 테스트
    print("\n" + "="*80)
    print("🔄 Testing Hybrid Search (all document types)")
    print("="*80)

    query = "방화벽 로그 분석"
    print(f"\n🔍 Query: {query}")
    print("-"*60)

    results = rag_system.search_hybrid(query, top_k=5)
    if results:
        print(f"📄 Top {len(results)} results from all types:")
        for i, (file_path, doc_data, score) in enumerate(results, 1):
            doc_type = doc_data.get('doc_type', 'unknown')
            print(f"  {i}. [{score:.4f}] [{doc_type}] {file_path}")
    else:
        print("  No results found")
