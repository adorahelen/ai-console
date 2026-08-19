#!/usr/bin/env python3
"""
BGE-M3 임베딩 어댑터 모듈
BGE-M3 모델을 사용하여 3가지 타입의 벡터(dense, sparse, colbert)를 생성
"""

import os
import hashlib
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from config_utils import qdrant_collection
from pathlib import Path
import logging
import warnings

# Qdrant 클라이언트 추가
try:
    from qdrant_client import QdrantClient, models
    from qdrant_client.http.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    print("⚠️ Qdrant 클라이언트를 사용할 수 없습니다. pip install qdrant-client로 설치하세요.")

# 토크나이저 경고 억제
warnings.filterwarnings("ignore", message=".*fast tokenizer.*__call__.*")
warnings.filterwarnings("ignore", message=".*XLMRobertaTokenizerFast.*")

# 진행바 관련 환경변수 설정 (조용한 모드)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TQDM_DISABLE'] = '1'  # tqdm 전체 비활성화
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'  # HuggingFace 진행바 비활성화

# 로깅 설정 (더욱 조용하게)
logging.getLogger('FlagEmbedding').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.WARNING)  # HTTP 요청 로그 비활성화
logging.getLogger('httpcore').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# FlagEmbedding import

try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    logger.warning("FlagEmbedding 라이브러리가 설치되지 않았습니다.")

# BGE 벡터 데이터베이스 저장을 위한 imports
import pickle
import json
import time
import re

# -----------------------------
# Command/Function doc branding
# -----------------------------
_CMD_DOC_PAT = re.compile(
    r"^query-(?P<name>.+?)-(?P<kind>command|function)\.ya?ml$",
    re.IGNORECASE
)

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

    if kind == "command":
        return {"mode_brand": "lp_command_ref", "command_name": name}
    else:
        return {"mode_brand": "lp_function_ref", "func_name": name}

# -----------------------------
# Security ID patterns
# -----------------------------
_ID_PATTERNS = {
    "capec": re.compile(r"\bCAPEC[\s\-_:]*0*(\d{1,5})\b", re.IGNORECASE),
    "cwe":   re.compile(r"\bCWE[\s\-_:]*0*(\d{1,6})\b", re.IGNORECASE),
    "cve":   re.compile(r"\bCVE[\s\-_:]*((?:19|20)\d{2}[\-_:]*\d{3,7})\b", re.IGNORECASE),
    "attack_t": re.compile(r"\bT(\d{4})(?:\.(\d{3}))?\b", re.IGNORECASE),
    "attack_g": re.compile(r"\bG(\d{4})\b", re.IGNORECASE),
    "attack_s": re.compile(r"\bS(\d{4})\b", re.IGNORECASE),
}

def parse_security_ids(text: str) -> Dict[str, Any]:
    """
    CAPEC/CWE/CVE/ATT&CK 식별자를 텍스트에서 추출해 정규화.
    """
    if not text:
        return {
            "capec_ids": [], "cwe_ids": [], "cve_ids": [],
            "attack_technique_ids": [], "attack_group_ids": [], "attack_software_ids": [],
            "norm_ids": []
        }

    capec_ids = sorted({int(m.group(1)) for m in _ID_PATTERNS["capec"].finditer(text)})
    cwe_ids   = sorted({int(m.group(1)) for m in _ID_PATTERNS["cwe"].finditer(text)})

    cve_ids = set()
    for m in _ID_PATTERNS["cve"].finditer(text):
        s = m.group(1).replace("_", "-").replace(":", "-")
        parts = s.split("-")
        if len(parts) >= 2:
            cve_ids.add(f"CVE-{parts[0]}-{parts[1]}")
    cve_ids = sorted(cve_ids)

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

# FAISS 라이브러리 import
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS 라이브러리가 설치되지 않았습니다. pip install faiss-cpu")


class BGEFaissAdapter:
    """
    BGE-M3의 3가지 벡터를 FAISS + 별도 저장으로 분리 관리
    Dense/ColBERT: FAISS IndexFlatIP (빠른 ANN 검색)
    Sparse: Dict 별도 저장 (수동 계산)
    """

    def __init__(self, dim=1024, enable_faiss=True):
        """
        BGE FAISS 어댑터 초기화

        Args:
            dim: 벡터 차원 (BGE-M3는 1024차원)
            enable_faiss: FAISS 인덱스 생성 여부 (기본값: True)
        """
        self.dim = dim
        self.enable_faiss = enable_faiss

        if enable_faiss and not FAISS_AVAILABLE:
            raise ImportError("FAISS 라이브러리가 필요합니다: pip install faiss-cpu")

        # FAISS 인덱스 (enable_faiss=True일 때만 생성)
        if enable_faiss:
            # Dense용 FAISS 인덱스 (코사인 유사도)
            self.dense_index = faiss.IndexFlatIP(dim)
            # ColBERT용 FAISS 인덱스 (코사인 유사도)
            self.colbert_index = faiss.IndexFlatIP(dim)
        else:
            self.dense_index = None
            self.colbert_index = None

        # Sparse 벡터는 별도 저장 (FAISS 미지원)
        self.sparse_vectors = {}  # {doc_id: sparse_dict}

        # 메타데이터 저장
        self.file_keys = []  # 파일명 리스트 (기존 패턴 따름)
        self.metadata = []  # 문서 정보
        self.doc_count = 0

        faiss_status = "활성화" if enable_faiss else "비활성화"
        logger.info(f"🏗️ BGE FAISS 어댑터 초기화: {dim}차원 (FAISS {faiss_status})")

    def add_bge_vectors(self, embeddings_data):
        """
        BGE 임베딩 데이터를 FAISS 인덱스에 추가

        Args:
            embeddings_data: BGE 임베딩 데이터 리스트
        """
        if not embeddings_data:
            logger.warning("⚠️ 임베딩 데이터가 비어있습니다")
            return

        # FAISS가 비활성화된 경우 메타데이터만 저장
        if not self.enable_faiss:
            logger.info("⚠️ FAISS가 비활성화됨 - 메타데이터만 저장")
            for i, data in enumerate(embeddings_data):
                # Sparse 벡터는 별도 저장
                doc_id = self.doc_count + i
                embedding = data['embedding']
                self.sparse_vectors[doc_id] = embedding.get('sparse', {})

                # 파일 키와 메타데이터 저장
                file_name = data.get('name', '')
                self.file_keys.append(file_name)

                self.metadata.append({
                    'name': file_name,
                    'content': data.get('content', ''),
                    'source': data.get('source', ''),
                    'type': data.get('type', 'qna'),
                    'description': data.get('description', ''),
                    'token_count': data.get('token_count', 0),
                    'tags': data.get('tags'),
                    'guid': data.get('guid')
                })

            self.doc_count += len(embeddings_data)
            logger.info(f"✅ 메타데이터만 저장: {len(embeddings_data)}개 문서 (총 {self.doc_count}개)")
            return

        dense_batch = []
        colbert_batch = []

        for i, data in enumerate(embeddings_data):
            embedding = data['embedding']

            # 1. Dense 벡터 준비
            dense_vec = np.array(embedding.get('dense', []), dtype=np.float32)
            if len(dense_vec) != self.dim:
                logger.warning(f"⚠️ Dense 벡터 차원 불일치: {len(dense_vec)} != {self.dim}")
                dense_vec = np.zeros(self.dim, dtype=np.float32)
            dense_batch.append(dense_vec)

            # 2. ColBERT 벡터 준비 (평균 풀링)
            colbert_vecs = embedding.get('colbert', [])
            if len(colbert_vecs) > 0:
                colbert_array = np.array(colbert_vecs)
                if len(colbert_array.shape) == 2:  # (seq_len, dim)
                    colbert_avg = np.mean(colbert_array, axis=0)
                else:  # 이미 평균된 1차원 벡터
                    colbert_avg = colbert_array
            else:
                colbert_avg = np.zeros(self.dim, dtype=np.float32)

            if len(colbert_avg) != self.dim:
                logger.warning(f"⚠️ ColBERT 벡터 차원 불일치: {len(colbert_avg)} != {self.dim}")
                colbert_avg = np.zeros(self.dim, dtype=np.float32)
            colbert_batch.append(colbert_avg.astype(np.float32))

            # 3. Sparse 벡터는 별도 저장
            doc_id = self.doc_count + i
            self.sparse_vectors[doc_id] = embedding.get('sparse', {})

            # 4. 파일 키와 메타데이터 저장
            file_name = data.get('name', '')
            self.file_keys.append(file_name)

            self.metadata.append({
                'name': file_name,
                'content': data.get('content', ''),
                'source': data.get('source', ''),
                'type': data.get('type', 'qna'),
                'description': data.get('description', ''),
                'token_count': data.get('token_count', 0),
                'tags': data.get('tags'),
                'guid': data.get('guid')
            })

        # FAISS에 벡터 추가 (정규화 후)
        dense_matrix = np.vstack(dense_batch)
        colbert_matrix = np.vstack(colbert_batch)

        # L2 정규화 (코사인 유사도용)
        faiss.normalize_L2(dense_matrix)
        faiss.normalize_L2(colbert_matrix)

        # FAISS 인덱스에 추가
        self.dense_index.add(dense_matrix)
        self.colbert_index.add(colbert_matrix)
        self.doc_count += len(embeddings_data)

        logger.info(f"✅ FAISS 인덱스 추가: {len(embeddings_data)}개 문서 (총 {self.doc_count}개)")

    def hybrid_search(self, query_embedding, top_k=5, weights=None):
        """
        BGE 하이브리드 검색 (Dense + Sparse + ColBERT)

        Args:
            query_embedding: BGE 쿼리 임베딩 딕셔너리
            top_k: 반환할 결과 수
            weights: 벡터별 가중치

        Returns:
            (source, content, score) 튜플 리스트
        """
        if weights is None:
            weights = {'dense': 0.4, 'sparse': 0.3, 'colbert': 0.3}

        if self.doc_count == 0:
            logger.warning("⚠️ 인덱스에 문서가 없습니다")
            return []

        # FAISS가 비활성화된 경우 Sparse만 사용
        if not self.enable_faiss:
            logger.info("⚠️ FAISS가 비활성화됨 - Sparse 검색만 수행")
            return self._sparse_only_search(query_embedding, top_k)

        try:
            # 1. Dense FAISS 검색
            query_dense = np.array([query_embedding.get('dense', [])], dtype=np.float32)
            faiss.normalize_L2(query_dense)

            search_k = min(top_k * 3, self.doc_count)  # 후보군 확대
            dense_scores, dense_ids = self.dense_index.search(query_dense, search_k)
            dense_scores = dense_scores[0]  # 배치 차원 제거
            dense_ids = dense_ids[0]

            # 2. ColBERT FAISS 검색
            query_colbert = query_embedding.get('colbert', [])
            if len(query_colbert) > 0:
                if len(np.array(query_colbert).shape) == 2:  # (seq_len, dim)
                    query_colbert_avg = np.mean(query_colbert, axis=0)
                else:
                    query_colbert_avg = query_colbert
            else:
                query_colbert_avg = np.zeros(self.dim)

            query_colbert_norm = np.array([query_colbert_avg], dtype=np.float32)
            faiss.normalize_L2(query_colbert_norm)

            colbert_scores, colbert_ids = self.colbert_index.search(query_colbert_norm, search_k)
            colbert_scores = colbert_scores[0]
            colbert_ids = colbert_ids[0]

            # 3. 모든 후보 문서 ID 수집
            candidate_ids = set(dense_ids[dense_ids >= 0]) | set(colbert_ids[colbert_ids >= 0])

            # 4. 하이브리드 점수 계산
            final_results = []
            query_sparse = query_embedding.get('sparse', {})

            for doc_id in candidate_ids:
                if doc_id >= len(self.metadata):
                    continue

                # Dense 점수
                dense_idx = np.where(dense_ids == doc_id)[0]
                dense_score = dense_scores[dense_idx[0]] if len(dense_idx) > 0 else 0.0

                # ColBERT 점수
                colbert_idx = np.where(colbert_ids == doc_id)[0]
                colbert_score = colbert_scores[colbert_idx[0]] if len(colbert_idx) > 0 else 0.0

                # Sparse 점수 (수동 계산)
                doc_sparse = self.sparse_vectors.get(int(doc_id), {})
                sparse_score = self._calculate_sparse_similarity(query_sparse, doc_sparse)

                # 최종 하이브리드 점수
                hybrid_score = (
                    float(dense_score) * weights['dense'] +
                    float(sparse_score) * weights['sparse'] +
                    float(colbert_score) * weights['colbert']
                )

                metadata = self.metadata[int(doc_id)]
                final_results.append((
                    hybrid_score,
                    int(doc_id),
                    metadata
                ))

            # 점수 기준 정렬 (점수는 첫 번째 요소)
            final_results.sort(key=lambda x: x[0], reverse=True)
            return final_results[:top_k]

        except Exception as e:
            logger.error(f"❌ BGE FAISS 하이브리드 검색 실패: {e}")
            return []

    def _calculate_sparse_similarity(self, query_sparse: Dict, doc_sparse: Dict) -> float:
        """Sparse 벡터 유사도 계산 (코사인 유사도)"""
        if not query_sparse or not doc_sparse:
            return 0.0

        # 공통 키워드만 계산
        common_keys = set(query_sparse.keys()) & set(doc_sparse.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(query_sparse[key] * doc_sparse[key] for key in common_keys)

        query_norm = sum(v**2 for v in query_sparse.values()) ** 0.5
        doc_norm = sum(v**2 for v in doc_sparse.values()) ** 0.5

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        return dot_product / (query_norm * doc_norm)

    def _sparse_only_search(self, query_embedding, top_k=5):
        """
        FAISS 비활성화 시 Sparse 벡터만 사용한 검색

        Args:
            query_embedding: BGE 쿼리 임베딩 딕셔너리
            top_k: 반환할 결과 수

        Returns:
            (score, doc_id, metadata) 튜플 리스트
        """
        query_sparse = query_embedding.get('sparse', {})
        if not query_sparse:
            logger.warning("⚠️ 쿼리에 Sparse 벡터가 없습니다")
            return []

        results = []
        for doc_id in range(self.doc_count):
            if doc_id >= len(self.metadata):
                continue

            # Sparse 유사도만 계산
            doc_sparse = self.sparse_vectors.get(doc_id, {})
            sparse_score = self._calculate_sparse_similarity(query_sparse, doc_sparse)

            if sparse_score > 0:  # 유사도가 0보다 큰 경우만
                metadata = self.metadata[doc_id]
                results.append((sparse_score, doc_id, metadata))

        # 스코어 기준 정렬
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def get_stats(self) -> Dict:
        """인덱스 통계 정보 반환"""
        stats = {
            'total_docs': self.doc_count,
            'sparse_vectors_count': len(self.sparse_vectors),
            'metadata_count': len(self.metadata),
            'faiss_enabled': self.enable_faiss
        }

        if self.enable_faiss and self.dense_index and self.colbert_index:
            stats.update({
                'dense_index_size': self.dense_index.ntotal,
                'colbert_index_size': self.colbert_index.ntotal
            })
        else:
            stats.update({
                'dense_index_size': 0,
                'colbert_index_size': 0
            })

        return stats


class BGEEmbeddingAdapter:
    """
    BGE-M3 모델을 사용한 임베딩 생성 어댑터
    dense, sparse, colbert 3가지 벡터를 생성하고 하나의 구조체로 패킹
    """

    def __init__(self, model_path: str = "/data/models/bge-m3", use_fp16: bool = True,
                 config_manager=None, db_manager=None, auto_save_to_db=True,
                 existing_model=None):
        """
        BGE-M3 모델 초기화

        Args:
            model_path: BGE-M3 모델 경로
            use_fp16: FP16 사용 여부 (GPU 메모리 절약)
            config_manager: 설정 관리자 (BGE Vector DB용)
            db_manager: DB 매니저 (BGE Vector DB용)
            auto_save_to_db: 임베딩 후 자동 DB 저장 여부
            existing_model: 이미 로드된 BGEM3FlagModel 인스턴스 (재사용)
        """
        if not BGE_AVAILABLE:
            raise ImportError("FlagEmbedding 라이브러리를 설치해주세요: pip install FlagEmbedding")

        self.model_path = model_path
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 기본 가중치 설정 (하이브리드 검색용)
        self.default_weights = {
            'dense': 0.40,
            'sparse': 0.30,
            'colbert': 0.30
        }

        # 기존 모델 재사용
        if existing_model is not None:
            self.model = existing_model
            logger.info("✅ BGE-M3 기존 모델 재사용 (GPU 메모리 절약)")
            return

        # 모델 경로 확인
        if not os.path.exists(model_path):
            logger.warning(f"모델 경로를 찾을 수 없습니다: {model_path}")
            logger.info("HuggingFace에서 모델을 다운로드합니다...")
            model_path = 'BAAI/bge-m3'  # HuggingFace 모델명

        logger.info(f"🚀 BGE-M3 모델 로드 중: {model_path}")
        logger.info(f"   디바이스: {self.device}, FP16: {self.use_fp16}")

        try:
            # BGE-M3 모델 초기화
            self.model = BGEM3FlagModel(
                model_path,
                use_fp16=self.use_fp16,
                device=self.device
            )
            logger.info("✅ BGE-M3 모델 로드 완료")

        except Exception as e:
            import traceback
            logger.error(f"❌ BGE-M3 모델 로드 실패: {e}")
            logger.error(f"상세 오류 정보:\n{traceback.format_exc()}")
            raise

    def generate_embedding(self, text: str) -> Dict[str, Any]:
        """
        텍스트에서 BGE-M3 임베딩 생성 (3가지 벡터)

        Args:
            text: 임베딩을 생성할 텍스트

        Returns:
            packed_vector를 포함한 딕셔너리
        """
        if not text or not text.strip():
            raise ValueError("빈 텍스트는 임베딩을 생성할 수 없습니다")

        try:
            # BGE-M3 인코딩 (3가지 벡터 모두 생성)
            embeddings = self.model.encode(
                [text],  # 리스트로 전달
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
                max_length=8192  # BGE-M3 최대 길이
            )

            # Dense 벡터 정규화 (리스트의 첫 번째 요소)
            dense_vector = embeddings['dense_vecs'][0] if isinstance(embeddings['dense_vecs'], (list, np.ndarray)) else embeddings['dense_vecs']
            if isinstance(dense_vector, np.ndarray):
                dense_vector = dense_vector.flatten()
                # L2 정규화
                norm = np.linalg.norm(dense_vector)
                if norm > 0:
                    dense_vector = dense_vector / norm

            # Sparse 벡터 처리 (딕셔너리 형태, 리스트의 첫 번째 요소)
            sparse_vector = embeddings['lexical_weights'][0] if isinstance(embeddings['lexical_weights'], list) else embeddings['lexical_weights']
            if isinstance(sparse_vector, dict):
                # 값 정규화 (옵션)
                max_weight = max(sparse_vector.values()) if sparse_vector else 1.0
                if max_weight > 0:
                    sparse_vector = {k: v/max_weight for k, v in sparse_vector.items()}

            # ColBERT 벡터 처리 (리스트의 첫 번째 요소)
            colbert_vectors = (
                embeddings["colbert_vecs"][0]
                if isinstance(embeddings["colbert_vecs"], (list, np.ndarray))
                else embeddings["colbert_vecs"]
            )

            # ✅ 문서 저장용: 2D 그대로 유지 (SEQ x 1024)
            # Qdrant multivector 제한: 총 요소 수 < 1,048,576 (= 1024 tokens × 1024 dim)
            MAX_COLBERT_TOKENS = 512  # 안전 마진 확보 (512 × 1024 = 524,288 요소)
            if isinstance(colbert_vectors, np.ndarray):
                colbert_matrix = colbert_vectors.astype(np.float32)

                # 토큰 수 제한: Qdrant multivector 요소 제한 초과 방지
                if colbert_matrix.shape[0] > MAX_COLBERT_TOKENS:
                    logger.info(f"ColBERT 토큰 수 제한: {colbert_matrix.shape[0]} → {MAX_COLBERT_TOKENS}")
                    colbert_matrix = colbert_matrix[:MAX_COLBERT_TOKENS]

                # 각 토큰 벡터 정규화 (MAX_SIM 안정화에 도움)
                norms = np.linalg.norm(colbert_matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                colbert_matrix = colbert_matrix / norms

                colbert_for_store = colbert_matrix.tolist()

            else:
                # list인 경우: 2D인지 확인해서 1D면 2D로 감싸기
                colbert_for_store = colbert_vectors
                if isinstance(colbert_for_store, list) and len(colbert_for_store) > 0 and not isinstance(colbert_for_store[0], list):
                    colbert_for_store = [colbert_for_store]

            # ✅ 안전장치: 빈 값이면 최소 형태 유지(선택)
            if not colbert_for_store:
                colbert_for_store = [[0.0] * 1024]

            # 3개 벡터를 하나의 구조체로 패킹
            packed_vector = {
                "type": "bge_m3",
                "dense": dense_vector.tolist() if isinstance(dense_vector, np.ndarray) else dense_vector,
                "sparse": sparse_vector,
                # ✅ 여기! 평균 풀링된 colbert_vector가 아니라, 2D인 colbert_for_store 저장
                "colbert": colbert_for_store,
                "weights": self.default_weights,
                "metadata": {
                    "model": "bge-m3",
                    "device": self.device,
                    "fp16": self.use_fp16,
                    "text_length": len(text),
                    "colbert_is_2d": True,  # 디버깅용(선택)
                    "colbert_seq_len": len(colbert_for_store) if isinstance(colbert_for_store, list) else 0,
                },
            }

            # 토큰 수 추정 (근사치)
            token_count = len(text.split()) * 1.3  # 평균적으로 단어당 1.3토큰

            # OpenAI API 형식과 호환되도록 반환
            return {
                "data": [{"embedding": packed_vector}],
                "usage": {"total_tokens": int(token_count)},
                "model": "bge-m3"
            }

        except Exception as e:
            logger.error(f"❌ BGE-M3 임베딩 생성 실패: {e}")
            raise

    def generate_embedding_batch(
        self,
        texts: List[str],
        *,
        max_colbert_tokens: int = 192,     # 비용 제어용 (128~256 권장)
        normalize_colbert_tokens: bool = True,
        normalize_sparse_by_max: bool = True
    ) -> List[Dict[str, Any]]:
        """
        여러 텍스트에서 BGE-M3 임베딩 배치 생성
        - dense: 1D(1024) L2 정규화
        - sparse: dict(token_id->weight) (옵션: max로 정규화)
        - colbert: 2D(SEQ x 1024) 그대로 저장 (옵션: 토큰별 L2 정규화 + 토큰 수 제한)
        """
        if not texts:
            return []

        try:
            batch_embeddings = self.model.encode(
                texts,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=True,
                max_length=8192,
                batch_size=16
            )

            results: List[Dict[str, Any]] = []

            dense_vecs = batch_embeddings.get("dense_vecs", [])
            sparse_vecs = batch_embeddings.get("lexical_weights", [])
            colbert_vecs = batch_embeddings.get("colbert_vecs", [])

            for i, text in enumerate(texts):
                # -------------------------
                # Dense (1D, L2 normalize)
                # -------------------------
                dense_vector = dense_vecs[i] if i < len(dense_vecs) else None
                if isinstance(dense_vector, np.ndarray):
                    dense_vector = dense_vector.astype(np.float32).flatten()
                    n = np.linalg.norm(dense_vector)
                    if n > 0:
                        dense_vector = dense_vector / n
                    dense_out = dense_vector.tolist()
                elif isinstance(dense_vector, list):
                    dv = np.array(dense_vector, dtype=np.float32).flatten()
                    n = np.linalg.norm(dv)
                    if n > 0:
                        dv = dv / n
                    dense_out = dv.tolist()
                else:
                    dense_out = [0.0] * 1024  # fallback

                # -------------------------
                # Sparse (dict)
                # -------------------------
                sparse_vector = sparse_vecs[i] if i < len(sparse_vecs) else {}
                if isinstance(sparse_vector, dict) and sparse_vector and normalize_sparse_by_max:
                    mx = max(float(v.item()) if hasattr(v, "item") else float(v) for v in sparse_vector.values())
                    if mx > 0:
                        sparse_vector = {
                            int(k): (float(v.item()) if hasattr(v, "item") else float(v)) / mx
                            for k, v in sparse_vector.items()
                        }
                    else:
                        sparse_vector = {int(k): float(v.item()) if hasattr(v, "item") else float(v) for k, v in sparse_vector.items()}
                elif isinstance(sparse_vector, dict):
                    # 타입 정리만
                    sparse_vector = {int(k): float(v.item()) if hasattr(v, "item") else float(v) for k, v in sparse_vector.items()}
                else:
                    sparse_vector = {}

                # -------------------------
                # ColBERT (2D 유지: SEQ x 1024)
                # -------------------------
                colbert_vectors = colbert_vecs[i] if i < len(colbert_vecs) else None

                # 기대: np.ndarray shape = (seq_len, 1024)
                if isinstance(colbert_vectors, np.ndarray):
                    cv = colbert_vectors.astype(np.float32)

                    # 1D(1024)로 들어오면: 현재 파이프라인이 아직 평균풀링을 하고 있다는 의미
                    # 이 경우는 "멀티벡터 colbert"로 쓰면 위험하므로 명확히 처리
                    if cv.ndim == 1:
                        # 강제로 2D로 감싸는 것은 MAX_SIM에서 왜곡을 유발할 수 있으므로 비권장
                        # 운영에서는 아래처럼 예외를 내거나, colbert rerank를 비활성화하는 라우팅이 안전합니다.
                        # 여기서는 저장용 포맷을 지키기 위해 1토큰짜리 2D로 만들되, metadata로 표시합니다.
                        cv = cv.reshape(1, -1)

                    # 토큰 수 제한
                    if max_colbert_tokens and cv.shape[0] > max_colbert_tokens:
                        cv = cv[:max_colbert_tokens]

                    # 토큰별 정규화
                    if normalize_colbert_tokens and cv.shape[0] > 0:
                        norms = np.linalg.norm(cv, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        cv = cv / norms

                    colbert_out = cv.tolist()
                    colbert_seq_len = int(cv.shape[0])
                    colbert_dim = int(cv.shape[1]) if cv.ndim == 2 else 0

                elif isinstance(colbert_vectors, list) and colbert_vectors:
                    # list[list[float]] 기대
                    if isinstance(colbert_vectors[0], list):
                        cv = np.array(colbert_vectors, dtype=np.float32)
                        if max_colbert_tokens and cv.shape[0] > max_colbert_tokens:
                            cv = cv[:max_colbert_tokens]
                        if normalize_colbert_tokens and cv.shape[0] > 0:
                            norms = np.linalg.norm(cv, axis=1, keepdims=True)
                            norms[norms == 0] = 1.0
                            cv = cv / norms
                        colbert_out = cv.tolist()
                        colbert_seq_len = int(cv.shape[0])
                        colbert_dim = int(cv.shape[1])
                    else:
                        # 1D list(1024)로 온 경우
                        cv = np.array(colbert_vectors, dtype=np.float32).reshape(1, -1)
                        if normalize_colbert_tokens:
                            n = np.linalg.norm(cv, axis=1, keepdims=True)
                            n[n == 0] = 1.0
                            cv = cv / n
                        colbert_out = cv.tolist()
                        colbert_seq_len = 1
                        colbert_dim = int(cv.shape[1])
                else:
                    # fallback: "2D 포맷" 유지
                    colbert_out = [[0.0] * 1024]
                    colbert_seq_len = 1
                    colbert_dim = 1024

                packed_vector = {
                    "type": "bge_m3",
                    "dense": dense_out,
                    "sparse": sparse_vector,
                    "colbert": colbert_out,  # ✅ 2D로 저장
                    "weights": self.default_weights,
                    "metadata": {
                        "model": "bge-m3",
                        "device": self.device,
                        "fp16": self.use_fp16,
                        "text_length": len(text),
                        "colbert_is_2d": True,
                        "colbert_seq_len": colbert_seq_len,
                        "colbert_dim": colbert_dim,
                    }
                }

                token_count = int(len(text.split()) * 1.3)
                results.append({
                    "data": [{"embedding": packed_vector}],
                    "usage": {"total_tokens": token_count},
                    "model": "bge-m3"
                })

            return results

        except Exception as e:
            logger.error(f"❌ BGE-M3 배치 임베딩 생성 실패: {e}")
            raise

    def update_weights(self, dense_weight: float = 0.4, sparse_weight: float = 0.3, colbert_weight: float = 0.3):
        """
        하이브리드 검색 가중치 업데이트

        Args:
            dense_weight: Dense 벡터 가중치
            sparse_weight: Sparse 벡터 가중치
            colbert_weight: ColBERT 벡터 가중치
        """
        total = dense_weight + sparse_weight + colbert_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"가중치 합이 1.0이 아닙니다: {total}. 정규화합니다.")
            dense_weight = dense_weight / total
            sparse_weight = sparse_weight / total
            colbert_weight = colbert_weight / total

        self.default_weights = {
            'dense': dense_weight,
            'sparse': sparse_weight,
            'colbert': colbert_weight
        }
        logger.info(f"✅ 가중치 업데이트: {self.default_weights}")

    def cleanup(self):
        """GPU 메모리 정리"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("✅ BGE-M3 모델 메모리 정리 완료")


# 테스트 코드
if __name__ == "__main__":
    import json

    print("BGE-M3 임베딩 어댑터 테스트")
    print("="*60)

    # 어댑터 초기화
    adapter = BGEEmbeddingAdapter()

    # 테스트 텍스트
    test_text = "BGE-M3는 dense, sparse, colbert 세 가지 방식의 임베딩을 동시에 제공하는 강력한 모델입니다."

    print(f"테스트 텍스트: {test_text}")
    print("-"*60)

    # 임베딩 생성
    result = adapter.generate_embedding(test_text)

    print(f"모델: {result.get('model')}")
    print(f"토큰 수: {result['usage']['total_tokens']}")

    embedding = result['data'][0]['embedding']
    print(f"\n임베딩 타입: {embedding['type']}")
    print(f"Dense 벡터 차원: {len(embedding['dense'])}")
    print(f"Sparse 벡터 키 개수: {len(embedding['sparse'])}")
    print(f"ColBERT 벡터 차원: {len(embedding['colbert'])}")
    print(f"가중치: {embedding['weights']}")
    print(f"메타데이터: {embedding['metadata']}")

    # 배치 테스트
    print("\n" + "="*60)
    print("배치 임베딩 테스트")
    test_texts = [
        "첫 번째 테스트 문장입니다.",
        "두 번째 테스트 문장입니다.",
        "세 번째 테스트 문장입니다."
    ]

    batch_results = adapter.generate_embedding_batch(test_texts)
    print(f"배치 크기: {len(batch_results)}")

    for i, result in enumerate(batch_results):
        embedding = result['data'][0]['embedding']
        print(f"\n텍스트 {i+1}:")
        print(f"  - Dense 차원: {len(embedding['dense'])}")
        print(f"  - Sparse 키: {len(embedding['sparse'])}")
        print(f"  - ColBERT 차원: {len(embedding['colbert'])}")

    # 정리
    adapter.cleanup()
    print("\n✅ 테스트 완료")


class BGEVectorDBManager:
    """
    BGE-M3 임베딩을 위한 MariaDB 벡터 인덱스 관리자
    BGE FAISS 인덱스를 ai_faiss_indices 테이블에 저장/로드
    """

    def __init__(self, config_manager, db_manager, enable_faiss=False):
        """
        BGE DB 관리자 초기화

        Args:
            config_manager: 설정 관리자
            db_manager: DB 매니저 인스턴스
            enable_faiss: FAISS 인덱스 생성 여부 (기본값: False)
        """
        self.config_manager = config_manager
        self.db_manager = db_manager
        self.enable_faiss = enable_faiss

        # Qdrant 클라이언트 초기화
        self.qdrant_client = None
        self.qdrant_enabled = False
        self._init_qdrant_client()

        faiss_status = "활성화" if enable_faiss else "비활성화"
        qdrant_status = "활성화" if self.qdrant_enabled else "비활성화"
        logger.info(f"🔗 BGE 벡터 DB 관리자 초기화 (FAISS {faiss_status}, Qdrant {qdrant_status})")

    def _init_qdrant_client(self):
        """Qdrant 클라이언트 초기화"""
        if not QDRANT_AVAILABLE:
            logger.warning("⚠️ Qdrant 클라이언트가 설치되지 않음")
            return

        try:
            # Config에서 Qdrant 설정 읽기
            use_server = self.config_manager.config.get("qdrant", "use_server", fallback="False").lower() == "true"

            if use_server:
                host = self.config_manager.config.get("qdrant", "host", fallback="localhost")
                port = int(self.config_manager.config.get("qdrant", "port", fallback="6333"))
                timeout = int(self.config_manager.config.get("qdrant", "timeout", fallback="120"))

                self.qdrant_client = QdrantClient(host=host, port=port, timeout=timeout)

                # 연결 테스트
                try:
                    collections = self.qdrant_client.get_collections()
                    self.qdrant_enabled = True
                    logger.info(f"✅ Qdrant 서버 연결 성공 ({host}:{port})")
                except Exception as e:
                    logger.warning(f"⚠️ Qdrant 서버 연결 실패: {e}")
                    self.qdrant_client = None

        except Exception as e:
            logger.error(f"❌ Qdrant 클라이언트 초기화 실패: {e}")

    def get_collection_count(self) -> int:
        """컬렉션의 현재 포인트 수 조회"""
        try:
            collection_info = self.qdrant_client.get_collection(qdrant_collection())
            return collection_info.points_count or 0
        except:
            return 0

    def upload_to_qdrant_server(self, sub_id: int, embeddings_data: List[Dict], start_id: int = None) -> int:
        """
        BGE 임베딩을 Qdrant 서버에 직접 업로드 (단일 컬렉션 'bge' 사용)

        Args:
            sub_id: 구독키 ID
            embeddings_data: BGE 임베딩 데이터 리스트
            start_id: 시작 ID (None이면 컬렉션에서 조회)

        Returns:
            업로드된 문서 수 (실패 시 -1)
        """
        if not self.qdrant_enabled:
            logger.warning("⚠️ Qdrant 서버가 활성화되지 않음")
            return -1

        try:
            # 단일 컬렉션 사용 - doc_type은 payload에 저장
            collection_name = qdrant_collection()

            # 각 문서에 doc_type 추가
            for data in embeddings_data:
                file_key = data.get('file_key', '')
                if '/' in file_key:
                    doc_type = file_key.split('/')[0]
                else:
                    doc_type = 'qna'  # 기본값

                # 지원되지 않는 타입은 qna로 분류
                if doc_type not in ['qna', 'action', 'plan']:
                    doc_type = 'qna'

                data['doc_type'] = doc_type

            # 시작 ID 확인 (외부에서 전달받지 않으면 컬렉션에서 조회)
            # UUID 기반 ID 모드: 각 문서에 guid가 있으면 해시로 고유 정수 ID 생성
            if start_id is None:
                start_id = 0
                try:
                    collection_info = self.qdrant_client.get_collection(collection_name)
                    start_id = collection_info.points_count or 0
                    logger.info(f"📊 {collection_name} 현재 문서 수: {start_id}, 시작 ID: {start_id}")
                except:
                    logger.info(f"📊 {collection_name} 새 컬렉션, 시작 ID: 0")

            uploaded_count = self._upload_docs_to_qdrant(sub_id, embeddings_data, start_id)
            if uploaded_count >= 0:
                logger.info(f"✅ Qdrant 업로드 성공 (sub_id={sub_id}): {uploaded_count}개 문서")
                return uploaded_count
            else:
                logger.error(f"❌ Qdrant 업로드 실패")
                return -1

        except Exception as e:
            logger.error(f"❌ Qdrant 업로드 중 오류: {e}")
            return -1

    def _upload_docs_to_qdrant(self, sub_id: int, docs: List[Dict], start_id: int = 0) -> bool:
        """문서를 Qdrant에 업로드 (단일 컬렉션 'bge' + sub_id/doc_type 필터 방식)

        Args:
            sub_id: 구독 ID (payload에 저장되어 검색 시 필터로 사용)
            docs: 업로드할 문서 리스트 (doc_type 포함)
            start_id: 시작 ID (청크 업로드 시 이전 청크의 마지막 ID + 1)
        """
        try:
            # 단일 컬렉션 사용
            collection_name = qdrant_collection()

            # 컬렉션 확인 후 없으면 생성 (기존 데이터 보존)
            try:
                self.qdrant_client.get_collection(collection_name)
                logger.info(f"📁 기존 컬렉션 사용: {collection_name}")
            except:
                # 컬렉션이 없으면 생성 (BGE-M3 멀티벡터 설정)
                DENSE_DIM = 1024
                COLBERT_DIM = 1024

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
                                comparator=models.MultiVectorComparator.MAX_SIM
                            ),
                            hnsw_config=models.HnswConfigDiff(m=16),  # ColBERT HNSW 인덱싱 활성화 (3-way RRF용)
                        ),
                    },
                    sparse_vectors_config={
                        "sparse": models.SparseVectorParams(),
                    },
                    optimizers_config=models.OptimizersConfigDiff(
                        default_segment_number=1,       # 세그먼트 1개로 통합 (소규모 데이터 최적)
                        indexing_threshold=3000,         # 소규모 데이터에 맞게 낮춤 (기본 20000)
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        full_scan_threshold=2000,        # HNSW 검색 일관성 확보
                    ),
                )
                logger.info(f"🆕 새 컬렉션 생성 (멀티벡터): {collection_name}")

            # 벡터 데이터 준비
            points = []
            valid_doc_count = 0
            for i, doc in enumerate(docs):
                # BGE 벡터에서 dense, sparse, colbert 벡터 추출
                embedding = doc.get('embedding', {})
                doc_type = doc.get('doc_type', 'qna')  # 문서 타입 (payload에 저장)

                # Dense 벡터
                dense_vector = None
                if isinstance(embedding, dict) and 'dense' in embedding:
                    dense_vector = embedding['dense']
                    if hasattr(dense_vector, 'tolist'):
                        dense_vector = dense_vector.tolist()
                else:
                    dense_vector = doc.get('dense_vector', [])

                if not dense_vector or len(dense_vector) != 1024:
                    logger.warning(f"⚠️ 잘못된 dense 벡터 차원: {len(dense_vector) if dense_vector else 0} (예상: 1024)")
                    continue

                # Sparse 벡터 (dict -> indices, values)
                sparse_dict = embedding.get('sparse', {})
                sparse_indices = []
                sparse_values = []
                if sparse_dict:
                    for idx, val in sparse_dict.items():
                        sparse_indices.append(int(idx))
                        sparse_values.append(float(val.item()) if hasattr(val, 'item') else float(val))

                # ColBERT 벡터 (multivector: List[List[float]])
                colbert_vector = embedding.get('colbert', [])
                if hasattr(colbert_vector, 'tolist'):
                    colbert_vector = colbert_vector.tolist()
                # 1D면 2D로 변환
                if colbert_vector and not isinstance(colbert_vector[0], list):
                    colbert_vector = [colbert_vector]
                # Qdrant multivector 요소 제한 (1,048,576) 초과 방지
                MAX_COLBERT_TOKENS_UPLOAD = 512
                if colbert_vector and len(colbert_vector) > MAX_COLBERT_TOKENS_UPLOAD:
                    logger.info(f"ColBERT 업로드 토큰 제한: {len(colbert_vector)} → {MAX_COLBERT_TOKENS_UPLOAD}")
                    colbert_vector = colbert_vector[:MAX_COLBERT_TOKENS_UPLOAD]

                # 메타데이터 처리 (aibot_rag_module_BGE.py의 로직 적용)
                raw_content = doc.get('text', '')
                yaml_content = raw_content.replace('¶', '\n')
                if not yaml_content.startswith('---'):
                    yaml_content = '---\n' + yaml_content

                # file_key에서 filename과 path 추출
                file_key = doc.get('file_key', '')
                filename = doc.get('filename', '') or (file_key.split('/')[-1] if file_key else '')
                path = doc.get('path', '') or file_key
                metadata = {
                    'name': doc.get('name', filename),
                    'type': doc_type,
                    'filename': filename,
                    'path': path,
                    'file_key': file_key,
                    'description': doc.get('description', ''),
                    'content': yaml_content,
                }

                # YAML 파싱으로 추가 메타데이터 추출 (guid, tags 포함)
                parsed_guid = None
                parsed_tags = None
                if '---' in yaml_content and len(yaml_content) > 10:
                    try:
                        import yaml
                        parsed_yaml = yaml.safe_load(yaml_content)
                        if isinstance(parsed_yaml, dict):
                            metadata.update({
                                'name': parsed_yaml.get('name', metadata['name']),
                                'type': parsed_yaml.get('type', metadata['type']),
                                'description': parsed_yaml.get('description', metadata['description'])
                            })
                            parsed_guid = parsed_yaml.get('guid')
                            parsed_tags = parsed_yaml.get('tags')
                    except:
                        pass

                # ID 생성: guid 기반 해시 고유 정수 ID (파일경로 기반 결정적 GUID)
                doc_guid = parsed_guid or doc.get('guid') or metadata.get('guid')
                if not doc_guid and file_key:
                    from aibot_prompts_functions import generate_file_guid
                    doc_guid = generate_file_guid(file_key)
                if doc_guid:
                    doc_id = int(hashlib.sha256(doc_guid.encode()).hexdigest()[:15], 16)
                else:
                    doc_id = start_id + valid_doc_count
                valid_doc_count += 1

                # Named 벡터 구성 (dense, sparse, colbert)
                named_vectors = {
                    "dense": dense_vector,
                }

                # ColBERT 벡터 추가 (있는 경우만)
                if colbert_vector:
                    named_vectors["colbert"] = colbert_vector

                # Sparse 벡터 추가 (있는 경우만)
                if sparse_indices and sparse_values:
                    named_vectors["sparse"] = models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )

                # payload 기본 필드
                payload = {
                    "sub_id": sub_id,  # 구독 ID (검색 시 필터로 사용)
                    "doc_type": doc_type,  # 문서 타입 (qna, action, plan)
                    "name": metadata['name'],
                    "type": metadata['type'],
                    "filename": metadata['filename'],
                    "path": metadata['path'],
                    "file_key": file_key,
                    "description": metadata['description'],
                    "content": metadata['content'],
                    "text": yaml_content,  # 원본 텍스트도 유지
                    "guid": doc_guid,  # 프롬프트 GUID (파일경로 기반 결정적 생성)
                    "enabled": True,  # 활성화 상태 (토글용)
                    "rev": 1,  # 리비전 번호 (싱크용, 업데이트 시 증가)
                    "content_hash": hashlib.sha256(yaml_content.encode()).hexdigest(),  # 콘텐츠 해시 (싱크 시 변경 감지)
                    "tags": parsed_tags,  # 태그 (검색 필터/가중치용, 추후 활용)
                }

                # ✅ command/function 문서면 mode_brand 및 이름 필드 주입 (파일명 기반)
                brand = extract_mode_brand_from_filename(
                    metadata.get("filename") or metadata.get("name") or metadata.get("path")
                )
                if brand:
                    payload.update(brand)

                # ✅ ID 추출: path + content + description를 합쳐서 파싱 (CVE, CAPEC, MITRE 등)
                id_ctx = f"{metadata.get('path', '')}\n{metadata.get('description', '')}\n{yaml_content}"
                ids = parse_security_ids(id_ctx)
                payload.update(ids)

                points.append(models.PointStruct(
                    id=doc_id,
                    vector=named_vectors,
                    payload=payload
                ))

            if points:
                # 벡터 업로드 (멀티벡터로 인해 배치 단위로 분할 - Qdrant 10MB 제한)
                batch_size = 2  # ColBERT 멀티벡터가 크므로 2개씩 업로드
                for batch_start in range(0, len(points), batch_size):
                    batch_points = points[batch_start:batch_start + batch_size]
                    self.qdrant_client.upsert(
                        collection_name=collection_name,
                        points=batch_points
                    )
                logger.info(f"📤 {collection_name}: {len(points)}개 벡터 업로드 완료")
                return len(points)  # 업로드된 개수 반환
            else:
                logger.warning(f"⚠️ 업로드할 벡터가 없음: {collection_name}")
                return 0

        except Exception as e:
            logger.error(f"❌ {collection_name} 업로드 실패: {e}")
            return -1

    def _check_faiss_index_exists(self, sub_id: int, doc_type: str) -> bool:
        """FAISS 인덱스가 DB에 존재하는지 확인"""
        try:
            index_name = f"bge_faiss_{doc_type}"
            # DB에서 해당 구독의 FAISS 인덱스 확인
            conn = self.db_manager.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM ai_faiss_indices WHERE subscription_id = %s AND index_name = %s",
                (sub_id, index_name)
            )
            count = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            return count > 0
        except Exception as e:
            logger.warning(f"⚠️ FAISS 인덱스 확인 실패 (구독: {sub_id}, 타입: {doc_type}): {e}")
            return False

    def get_bge_faiss_index_name(self, doc_type: str) -> str:
        """BGE FAISS용 인덱스명 생성 (기존 FAISS와 구분)"""
        return f"bge_faiss_{doc_type}"  # 예: bge_faiss_qna, bge_faiss_action, bge_faiss_plan

    def build_and_store_bge_faiss_index(self, sub_id: int, embeddings_data: List[Dict]) -> bool:
        """
        BGE 임베딩 데이터로 FAISS 인덱스 구축 후 DB 저장

        Args:
            sub_id: 구독키 ID
            embeddings_data: BGE 임베딩 데이터 리스트

        Returns:
            저장 성공 여부
        """
        # FAISS가 비활성화된 경우 건너뛰기
        if not self.enable_faiss:
            logger.info(f"⚠️ FAISS 인덱스 생성이 비활성화됨 - 구독키 {sub_id} BGE 데이터 건너뛰기")
            return True

        try:
            # 문서 타입별로 그룹화
            doc_groups = {'qna': [], 'action': [], 'plan': []}

            for data in embeddings_data:
                # file_key에서 문서 타입 추출 (예: action/file.yaml -> action)
                file_key = data.get('file_key', '')
                if '/' in file_key:
                    doc_type = file_key.split('/')[0]
                else:
                    doc_type = 'qna'  # 기본값

                # 지원되지 않는 타입은 qna로 분류
                if doc_type not in doc_groups:
                    doc_type = 'qna'

                doc_groups[doc_type].append(data)

            total_stored = 0

            # 각 문서 타입별로 FAISS 인덱스 구축 및 저장
            for doc_type, docs in doc_groups.items():
                if not docs:
                    continue

                success = self._build_and_store_faiss_by_type(sub_id, doc_type, docs)
                if success:
                    total_stored += len(docs)
                    logger.info(f"✅ BGE FAISS {doc_type} 인덱스 저장: {len(docs)}개 문서")

            logger.info(f"✅ BGE FAISS 인덱스 저장 완료: 구독키 {sub_id} - 총 {total_stored}개 문서")
            return True

        except Exception as e:
            logger.error(f"❌ BGE FAISS 인덱스 저장 실패: {e}")
            return False

    def _build_and_store_faiss_by_type(self, sub_id: int, doc_type: str, docs: List[Dict]) -> bool:
        """문서타입별 BGE FAISS 인덱스 구축 및 저장 (기존 FAISS 패턴 따름)"""
        try:
            # 벡터 데이터 수집 및 검증
            dense_vectors = []
            colbert_vectors = []
            sparse_vectors = {}
            file_keys = []
            metadata_list = []

            for i, doc in enumerate(docs):
                embedding = doc.get('embedding', {})
                dense_vec = embedding.get('dense', [])
                colbert_vec = embedding.get('colbert', [])
                sparse_vec = embedding.get('sparse', {})

                # 차원 검증
                if len(dense_vec) == 1024 and len(colbert_vec) == 1024:
                    dense_vectors.append(dense_vec)
                    colbert_vectors.append(colbert_vec)
                    sparse_vectors[i] = sparse_vec
                    file_keys.append(doc.get('name', ''))
                    metadata_list.append({
                        'source': doc.get('source', ''),
                        'content': doc.get('content', ''),
                        'guid': doc.get('guid')
                    })

            if not dense_vectors:
                logger.warning(f"BGE {doc_type} - 유효한 벡터가 없어서 인덱스를 생성하지 않습니다.")
                return True

            # numpy 배열로 변환 (기존 패턴 따름)
            import numpy as np
            dense_array = np.array(dense_vectors, dtype=np.float32)
            colbert_array = np.array(colbert_vectors, dtype=np.float32)

            # L2 정규화 (기존 패턴 따름)
            faiss.normalize_L2(dense_array)
            faiss.normalize_L2(colbert_array)

            # FAISS 인덱스 생성 (기존 패턴 따름)
            dense_index = faiss.IndexFlatIP(1024)
            colbert_index = faiss.IndexFlatIP(1024)

            dense_index.add(dense_array)
            colbert_index.add(colbert_array)

            # BGE FAISS 데이터 패킹 (기존 직렬화 패턴 따름)
            index_data = {
                'dense_index': pickle.dumps(faiss.serialize_index(dense_index)),
                'colbert_index': pickle.dumps(faiss.serialize_index(colbert_index)),
                'sparse_vectors': sparse_vectors,
                'file_keys': file_keys,
                'metadata': metadata_list,
                'doc_count': len(file_keys)
            }

            # 전체 데이터를 pickle로 직렬화
            pickled_data = pickle.dumps(index_data)

            # 메타데이터 구성 (기존 패턴 따름)
            metadata = {
                'index_type': 'BGEFaiss',
                'intent': doc_type,
                'total_docs': len(file_keys),
                'embedding_dim': 1024,
                'vector_types': ['dense', 'sparse', 'colbert'],
                'created_at': datetime.now().isoformat()
            }

            # FAISS 인덱스를 메모리에만 저장 (DB 저장 제거됨)
            logger.info(f"📀 BGE FAISS {doc_type} 인덱스 메모리 저장 완료")
            return True

        except Exception as e:
            logger.error(f"❌ BGE FAISS {doc_type} 인덱스 구축 실패: {e}")
            return False

    def load_bge_faiss_index(self, sub_id: int, doc_type: str) -> Optional[BGEFaissAdapter]:
        """
        DB에서 BGE FAISS 인덱스 로드 (기존 패턴 따름)

        Args:
            sub_id: 구독키 ID
            doc_type: 문서 타입

        Returns:
            BGEFaissAdapter 인스턴스 또는 None
        """
        try:
            with self.db_manager.get_connection() as connection:
                with connection.cursor() as cursor:
                    index_name = self.get_bge_faiss_index_name(doc_type)

                    cursor.execute("""
                        SELECT index_data, metadata FROM ai_faiss_indices
                        WHERE subscription_id = %s AND index_name = %s
                    """, (sub_id, index_name))

                    row = cursor.fetchone()
                    if not row:
                        logger.warning(f"⚠️ BGE FAISS 인덱스 없음: 구독키 {sub_id}, 타입 {doc_type}")
                        return None

                    # 직렬화된 데이터 복원
                    index_data = pickle.loads(row[0])  # index_data

                    # BGE FAISS 어댑터 복원
                    bge_adapter = BGEFaissAdapter()

                    # FAISS 인덱스 복원 (기존 패턴 따름)
                    bge_adapter.dense_index = faiss.deserialize_index(pickle.loads(index_data['dense_index']))
                    bge_adapter.colbert_index = faiss.deserialize_index(pickle.loads(index_data['colbert_index']))

                    # 기타 데이터 복원
                    bge_adapter.sparse_vectors = index_data['sparse_vectors']
                    bge_adapter.metadata = index_data['metadata']
                    bge_adapter.doc_count = index_data['doc_count']
                    bge_adapter.file_keys = index_data.get('file_keys', [])

                    logger.info(f"✅ BGE FAISS 인덱스 로드: {doc_type} - {bge_adapter.doc_count}개 문서")
                    return bge_adapter

        except Exception as e:
            logger.error(f"❌ BGE FAISS 인덱스 로드 실패 (구독키 {sub_id}, 타입 {doc_type}): {e}")
            return None

    def load_all_bge_faiss_indices(self, sub_id: int) -> Dict[str, BGEFaissAdapter]:
        """
        구독키별 모든 BGE FAISS 인덱스 로드

        Args:
            sub_id: 구독키 ID

        Returns:
            문서타입별 BGEFaissAdapter 딕셔너리
        """
        indices = {}

        for doc_type in ['qna', 'action', 'plan']:
            adapter = self.load_bge_faiss_index(sub_id, doc_type)
            if adapter:
                indices[doc_type] = adapter

        total_docs = sum(adapter.doc_count for adapter in indices.values())
        logger.info(f"✅ BGE FAISS 인덱스 전체 로드: 구독키 {sub_id} - {len(indices)}개 타입, {total_docs}개 문서")

        return indices

    def store_bge_embeddings(self, sub_id: int, embeddings_data: List[Dict], start_id: int = None) -> int:
        """
        BGE-M3 임베딩을 저장 (Qdrant 서버 우선, 실패시 FAISS)

        Args:
            sub_id: 구독키 ID
            embeddings_data: BGE 임베딩 데이터 리스트
            start_id: 시작 ID (None이면 컬렉션에서 조회)

        Returns:
            업로드된 문서 수 (실패 시 -1)
        """
        # 1. Qdrant 서버에 업로드 시도 (우선순위)
        if self.qdrant_enabled:
            logger.info(f"📤 Qdrant 서버에 BGE 임베딩 업로드 시도 (구독키: {sub_id}, start_id: {start_id})")
            uploaded_count = self.upload_to_qdrant_server(sub_id, embeddings_data, start_id)
            if uploaded_count >= 0:
                logger.info(f"✅ Qdrant 서버 업로드 완료 (구독키: {sub_id}): {uploaded_count}개")
                return uploaded_count
            else:
                logger.warning(f"⚠️ Qdrant 서버 업로드 실패, FAISS로 폴백 (구독키: {sub_id})")

        # 2. FAISS 인덱스로 폴백
        # BGE 모드에서 FAISS 가 비활성화된 상태에서 Qdrant 도 연결 안 되면
        # 어디에도 저장되지 않으므로 실패로 간주해야 함 (silent failure 방지)
        if not self.enable_faiss:
            logger.error(
                f"❌ Qdrant 미연결 + FAISS 비활성화 → 저장할 백엔드 없음 (구독키: {sub_id}). "
                f"Qdrant 서비스/설정 확인 필요: config.ini [qdrant] 섹션, "
                f"curl http://<host>:<port>/collections"
            )
            return -1

        logger.info(f"📁 FAISS 인덱스에 BGE 임베딩 저장 (구독키: {sub_id})")
        success = self.build_and_store_bge_faiss_index(sub_id, embeddings_data)
        return len(embeddings_data) if success else -1

    def update_document(self, sub_id: int, point_id: int, packed_vector: Dict, doc: Dict) -> bool:
        """
        기존 문서의 벡터와 payload를 업데이트

        Args:
            sub_id: 구독키 ID
            point_id: 업데이트할 point ID
            packed_vector: BGE 임베딩 (dense, sparse, colbert)
            doc: 문서 정보 (text, file_key 등)

        Returns:
            성공 여부
        """
        from qdrant_client import models
        import yaml

        try:
            collection_name = qdrant_collection()

            # 1. 벡터 추출 및 변환
            dense_vector = packed_vector.get('dense', [])
            if hasattr(dense_vector, 'tolist'):
                dense_vector = dense_vector.tolist()

            if not dense_vector or len(dense_vector) != 1024:
                logger.warning(f"⚠️ 잘못된 dense 벡터 차원: {len(dense_vector) if dense_vector else 0}")
                return False

            # Sparse 벡터
            sparse_dict = packed_vector.get('sparse', {})
            sparse_indices = []
            sparse_values = []
            if sparse_dict:
                for idx, val in sparse_dict.items():
                    sparse_indices.append(int(idx))
                    sparse_values.append(float(val.item()) if hasattr(val, 'item') else float(val))

            # ColBERT 벡터
            colbert_vector = packed_vector.get('colbert', [])
            if hasattr(colbert_vector, 'tolist'):
                colbert_vector = colbert_vector.tolist()
            if colbert_vector and not isinstance(colbert_vector[0], list):
                colbert_vector = [colbert_vector]
            # Qdrant multivector 요소 제한 초과 방지
            MAX_COLBERT_TOKENS_UPLOAD = 512
            if colbert_vector and len(colbert_vector) > MAX_COLBERT_TOKENS_UPLOAD:
                logger.info(f"ColBERT 단건 업로드 토큰 제한: {len(colbert_vector)} → {MAX_COLBERT_TOKENS_UPLOAD}")
                colbert_vector = colbert_vector[:MAX_COLBERT_TOKENS_UPLOAD]

            # 2. Named 벡터 구성
            named_vectors = {"dense": dense_vector}
            if colbert_vector:
                named_vectors["colbert"] = colbert_vector
            if sparse_indices and sparse_values:
                named_vectors["sparse"] = models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values
                )

            # 3. 메타데이터 처리
            raw_content = doc.get('text', '')
            yaml_content = raw_content.replace('¶', '\n')
            if yaml_content and not yaml_content.startswith('---'):
                yaml_content = '---\n' + yaml_content

            file_key = doc.get('file_key', '')
            filename = doc.get('filename', '') or (file_key.split('/')[-1] if file_key else '')
            path = doc.get('path', '') or file_key
            doc_type = doc.get('doc_type', 'qna')

            metadata = {
                'name': doc.get('name', filename),
                'type': doc_type,
                'filename': filename,
                'path': path,
                'file_key': file_key,
                'description': doc.get('description', ''),
                'content': yaml_content,
            }

            # YAML 파싱으로 추가 메타데이터 추출 (guid, tags 포함)
            parsed_guid = doc.get('guid')
            parsed_tags = doc.get('tags')
            if '---' in yaml_content and len(yaml_content) > 10:
                try:
                    parsed_yaml = yaml.safe_load(yaml_content)
                    if isinstance(parsed_yaml, dict):
                        metadata.update({
                            'name': parsed_yaml.get('name', metadata['name']),
                            'type': parsed_yaml.get('type', metadata['type']),
                            'description': parsed_yaml.get('description', metadata['description'])
                        })
                        if not parsed_guid:
                            parsed_guid = parsed_yaml.get('guid')
                        if not parsed_tags:
                            parsed_tags = parsed_yaml.get('tags')
                except:
                    pass

            # guid: YAML에 있으면 사용, 없으면 file_key 기반 결정적 생성
            doc_guid = parsed_guid
            if not doc_guid and file_key:
                from aibot_prompts_functions import generate_file_guid
                doc_guid = generate_file_guid(file_key)

            # 4. Payload 구성
            payload = {
                "sub_id": sub_id,
                "doc_type": doc_type,
                "name": metadata['name'],
                "type": metadata['type'],
                "filename": metadata['filename'],
                "path": metadata['path'],
                "file_key": file_key,
                "description": metadata['description'],
                "content": metadata['content'],
                "text": yaml_content,
                "guid": doc_guid,
                "enabled": doc.get('enabled', True),
                "rev": doc.get('rev', 1),  # 리비전 번호 (싱크용, 업데이트 시 증가)
                "content_hash": hashlib.sha256(yaml_content.encode()).hexdigest(),  # 콘텐츠 해시 (싱크 시 변경 감지)
                "tags": parsed_tags,  # 태그 (검색 필터/가중치용, 추후 활용)
            }

            # mode_brand 추출
            brand = extract_mode_brand_from_filename(
                metadata.get("filename") or metadata.get("name") or metadata.get("path")
            )
            if brand:
                payload.update(brand)

            # 보안 ID 추출 (CVE, CAPEC, MITRE 등)
            id_ctx = f"{metadata.get('path', '')}\n{metadata.get('description', '')}\n{yaml_content}"
            ids = parse_security_ids(id_ctx)
            payload.update(ids)

            # 5. Upsert로 업데이트
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=named_vectors,
                        payload=payload
                    )
                ]
            )

            logger.info(f"✅ 문서 업데이트 완료: point_id={point_id}, guid={doc_guid}")
            return True

        except Exception as e:
            logger.error(f"❌ 문서 업데이트 실패: {e}")
            return False

    def search_hybrid(self, sub_id: int, doc_type: str, query_embedding: Dict,
                     top_k: int = 5, weights: Dict = None) -> List[Tuple[str, str, float]]:
        """
        BGE-M3 하이브리드 검색 수행 (메모리 로드된 FAISS 인덱스 활용)

        Args:
            sub_id: 구독키 ID
            doc_type: 문서 타입
            query_embedding: 쿼리 BGE 임베딩
            top_k: 반환할 결과 수
            weights: 벡터별 가중치
        """
        try:
            # BGE FAISS 인덱스 로드
            bge_adapter = self.load_bge_faiss_index(sub_id, doc_type)
            if not bge_adapter:
                logger.warning(f"⚠️ BGE FAISS 인덱스 없음: 구독키 {sub_id}, 타입 {doc_type}")
                return []

            # FAISS 기반 하이브리드 검색 수행
            return bge_adapter.hybrid_search(query_embedding, top_k, weights)

        except Exception as e:
            logger.error(f"❌ BGE FAISS 하이브리드 검색 실패: {e}")
            return []

    def delete_bge_vectors(self, sub_id: int, doc_type: str = None) -> bool:
        """BGE FAISS 인덱스 삭제 (구독키별 또는 특정 문서타입)"""
        try:
            with self.db_manager.get_connection() as connection:
                with connection.cursor() as cursor:
                    if doc_type:
                        index_name = self.get_bge_faiss_index_name(doc_type)
                        cursor.execute("""
                            DELETE FROM ai_faiss_indices
                            WHERE subscription_id = %s AND index_name = %s
                        """, (sub_id, index_name))
                        logger.info(f"🗑️ BGE FAISS 인덱스 삭제: 구독키 {sub_id}, 타입 {doc_type}")
                    else:
                        cursor.execute("""
                            DELETE FROM ai_faiss_indices
                            WHERE subscription_id = %s AND index_name LIKE 'bge_faiss_%'
                        """, (sub_id,))
                        logger.info(f"🗑️ BGE FAISS 인덱스 삭제: 구독키 {sub_id} 전체")

                    connection.commit()
                    return True

        except Exception as e:
            logger.error(f"❌ BGE FAISS 인덱스 삭제 실패: {e}")
            return False

    def get_bge_stats(self) -> Dict[str, int]:
        """BGE FAISS 인덱스 통계 조회"""
        try:
            with self.db_manager.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT
                            subscription_id,
                            index_name,
                            total_docs
                        FROM ai_faiss_indices
                        WHERE index_name LIKE 'bge_faiss_%'
                        ORDER BY subscription_id, index_name
                    """)

                    rows = cursor.fetchall()
                    stats = {}
                    total = 0

                    for row in rows:
                        sub_id = row[0]  # subscription_id
                        index_name = row[1]  # index_name
                        doc_type = index_name.replace('bge_faiss_', '')
                        count = row[2] or 0  # total_docs

                        if sub_id not in stats:
                            stats[sub_id] = {}
                        stats[sub_id][doc_type] = count
                        total += count

                    stats['total'] = total
                    return stats

        except Exception as e:
            logger.error(f"❌ BGE FAISS 통계 조회 실패: {e}")
            return {}

    # 이전 메서드들 제거 (호환성을 위해 유지)
    def _store_bge_index_by_type(self, sub_id: int, doc_type: str, docs: List[Dict]) -> bool:
        """문서타입별 BGE 인덱스를 ai_faiss_indices에 저장"""
        try:
            with self.db_manager.get_connection() as connection:
                with connection.cursor() as cursor:
                    # BGE 벡터들을 하나의 구조체로 패킹
                    packed_data = {
                        'vectors': [],
                        'file_keys': [],
                        'metadata': []
                    }

                    for doc in docs:
                        embedding = doc['embedding']

                        # 3가지 벡터를 하나로 패킹
                        vector_pack = {
                            'dense': embedding.get('dense', []),
                            'sparse': embedding.get('sparse', {}),
                            'colbert': embedding.get('colbert', []),
                            'weights': embedding.get('weights', {})
                        }

                        packed_data['vectors'].append(vector_pack)
                        packed_data['file_keys'].append(doc.get('name', ''))
                        packed_data['metadata'].append({
                            'source': doc.get('source', ''),
                            'content': doc.get('content', ''),
                            'description': doc.get('description', ''),
                            'token_count': doc.get('token_count', 0),
                            'tags': doc.get('tags'),
                            'guid': doc.get('guid')
                        })

                    # 전체 데이터를 pickle로 직렬화
                    index_data = pickle.dumps(packed_data)

                    # 메타데이터 구성
                    metadata = {
                        'index_type': 'bge_m3',
                        'intent': doc_type,
                        'total_docs': len(docs),
                        'vector_type': 'hybrid',  # dense + sparse + colbert
                        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                    }

                    index_name = self.get_bge_index_name(doc_type)

                    # ai_faiss_indices에 UPSERT
                    cursor.execute("""
                        SELECT id FROM ai_faiss_indices
                        WHERE subscription_id = %s AND index_name = %s
                    """, (sub_id, index_name))

                    if cursor.fetchone():
                        # 업데이트
                        cursor.execute("""
                            UPDATE ai_faiss_indices
                            SET index_data = %s, metadata = %s, rev = rev + 1, updated_at = NOW()
                            WHERE subscription_id = %s AND index_name = %s
                        """, (index_data, json.dumps(metadata), sub_id, index_name))
                    else:
                        # 삽입
                        cursor.execute("""
                            INSERT INTO ai_faiss_indices
                            (subscription_id, index_name, index_data, metadata)
                            VALUES (%s, %s, %s, %s)
                        """, (sub_id, index_name, index_data, json.dumps(metadata)))

                    connection.commit()
                    return True

        except Exception as e:
            logger.error(f"❌ BGE {doc_type} 인덱스 저장 실패: {e}")
            return False

    def load_bge_vectors_by_subscription(self, sub_id: int) -> Dict[str, List[Dict]]:
        """
        구독키별 BGE 벡터를 ai_faiss_indices에서 로드 (문서타입별로 분류)

        Args:
            sub_id: 구독키 ID

        Returns:
            문서타입별 벡터 데이터 딕셔너리
        """
        try:
            with self.db_manager.get_connection() as connection:
                with connection.cursor() as cursor:
                    # BGE 인덱스들을 조회
                    cursor.execute("""
                        SELECT index_name, index_data, metadata
                        FROM ai_faiss_indices
                        WHERE subscription_id = %s AND index_name LIKE 'bge_%'
                    """, (sub_id,))

                    rows = cursor.fetchall()
                    vectors_by_type = {'qna': [], 'action': [], 'plan': []}

                    for row in rows:
                        try:
                            # 인덱스명에서 문서타입 추출 (bge_qna -> qna)
                            index_name = row[0]  # index_name
                            doc_type = index_name.replace('bge_', '')

                            if doc_type not in vectors_by_type:
                                continue

                            # 패킹된 데이터를 역직렬화
                            packed_data = pickle.loads(row[1])  # index_data
                            vectors = packed_data.get('vectors', [])
                            file_keys = packed_data.get('file_keys', [])
                            metadata_list = packed_data.get('metadata', [])

                            # 각 문서별로 벡터 데이터 복원
                            for i, vector_pack in enumerate(vectors):
                                if i < len(file_keys) and i < len(metadata_list):
                                    file_key = file_keys[i]
                                    doc_meta = metadata_list[i]

                                    vector_data = {
                                        'name': file_key,
                                        'source': doc_meta.get('source', ''),
                                        'content': doc_meta.get('content', ''),
                                        'description': doc_meta.get('description', ''),
                                        'dense': vector_pack.get('dense', []),
                                        'sparse': vector_pack.get('sparse', {}),
                                        'colbert': vector_pack.get('colbert', []),
                                        'weights': vector_pack.get('weights', {}),
                                        'token_count': doc_meta.get('token_count', 0),
                                        'tags': doc_meta.get('tags'),
                                        'guid': doc_meta.get('guid')
                                    }

                                    vectors_by_type[doc_type].append(vector_data)

                            logger.info(f"✅ {doc_type}: {len(vectors)}개 BGE 벡터 로드")

                        except Exception as unpack_error:
                            logger.warning(f"⚠️ BGE 인덱스 언패킹 실패 ({index_name}): {unpack_error}")
                            continue

                    total_count = sum(len(vecs) for vecs in vectors_by_type.values())
                    logger.info(f"✅ BGE 벡터 로드 완료: 구독키 {sub_id} - {total_count}개 문서")

                    return vectors_by_type

        except Exception as e:
            logger.error(f"❌ BGE 벡터 로드 실패 (구독키 {sub_id}): {e}")
            return {'qna': [], 'action': [], 'plan': []}

    def search_hybrid(self, sub_id: int, doc_type: str, query_embedding: Dict,
                     top_k: int = 5, weights: Dict = None) -> List[Tuple[str, str, float]]:
        """
        BGE-M3 하이브리드 검색 수행 (기존 ai_faiss_indices 활용)

        Args:
            sub_id: 구독키 ID
            doc_type: 문서 타입
            query_embedding: 쿼리 BGE 임베딩
            top_k: 반환할 결과 수
            weights: 벡터별 가중치
        """
        try:
            with self.db_manager.get_connection() as connection:
                with connection.cursor() as cursor:
                    # 기본 가중치 설정
                    if weights is None:
                        weights = {"dense": 0.4, "sparse": 0.3, "colbert": 0.3}

                    index_name = self.get_bge_index_name(doc_type)

                    # BGE 인덱스 로드
                    cursor.execute("""
                        SELECT index_data FROM ai_faiss_indices
                        WHERE subscription_id = %s AND index_name = %s
                    """, (sub_id, index_name))

                    row = cursor.fetchone()
                    if not row:
                        logger.warning(f"⚠️ BGE 인덱스 없음: 구독키 {sub_id}, 타입 {doc_type}")
                        return []

                    # 패킹된 데이터 역직렬화
                    packed_data = pickle.loads(row[0])  # row[0] = index_data
                    vectors = packed_data.get('vectors', [])
                    file_keys = packed_data.get('file_keys', [])
                    metadata_list = packed_data.get('metadata', [])

                    if not vectors:
                        logger.warning(f"⚠️ BGE 벡터 데이터 없음: 구독키 {sub_id}, 타입 {doc_type}")
                        return []

                    # 쿼리 벡터 추출
                    query_dense = np.array(query_embedding.get('dense', []))
                    query_sparse = query_embedding.get('sparse', {})
                    query_colbert = np.array(query_embedding.get('colbert', []))

                    results = []

                    for i, vector_pack in enumerate(vectors):
                        try:
                            if i >= len(file_keys) or i >= len(metadata_list):
                                continue

                            # 저장된 벡터 추출
                            doc_dense = np.array(vector_pack.get('dense', []))
                            doc_sparse = vector_pack.get('sparse', {})
                            doc_colbert = np.array(vector_pack.get('colbert', []))

                            # Dense 유사도 계산 (코사인)
                            dense_score = 0.0
                            if len(query_dense) > 0 and len(doc_dense) > 0:
                                dense_score = np.dot(query_dense, doc_dense) / (
                                    np.linalg.norm(query_dense) * np.linalg.norm(doc_dense)
                                )

                            # Sparse 유사도 계산
                            sparse_score = self._calculate_sparse_similarity(query_sparse, doc_sparse)

                            # ColBERT 유사도 계산 (코사인)
                            colbert_score = 0.0
                            if len(query_colbert) > 0 and len(doc_colbert) > 0:
                                colbert_score = np.dot(query_colbert, doc_colbert) / (
                                    np.linalg.norm(query_colbert) * np.linalg.norm(doc_colbert)
                                )

                            # 하이브리드 점수 계산
                            hybrid_score = (
                                dense_score * weights['dense'] +
                                sparse_score * weights['sparse'] +
                                colbert_score * weights['colbert']
                            )

                            # 결과 추가
                            doc_meta = metadata_list[i]
                            source = doc_meta.get('source', file_keys[i])
                            content = doc_meta.get('content', '')

                            results.append((source, content, float(hybrid_score)))

                        except Exception as doc_error:
                            logger.warning(f"⚠️ 문서 유사도 계산 실패 (인덱스 {i}): {doc_error}")
                            continue

                    # 점수 기준 내림차순 정렬 후 상위 k개 반환
                    results.sort(key=lambda x: x[2], reverse=True)
                    return results[:top_k]

        except Exception as e:
            logger.error(f"❌ BGE 하이브리드 검색 실패 (구독키 {sub_id}, 타입 {doc_type}): {e}")
            return []

    def _calculate_sparse_similarity(self, query_sparse: Dict, doc_sparse: Dict) -> float:
        """Sparse 벡터 유사도 계산 (코사인 유사도)"""
        if not query_sparse or not doc_sparse:
            return 0.0

        # 공통 키워드만 계산
        common_keys = set(query_sparse.keys()) & set(doc_sparse.keys())
        if not common_keys:
            return 0.0

        dot_product = sum(query_sparse[key] * doc_sparse[key] for key in common_keys)

        query_norm = sum(v**2 for v in query_sparse.values()) ** 0.5
        doc_norm = sum(v**2 for v in doc_sparse.values()) ** 0.5

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        return dot_product / (query_norm * doc_norm)

    def delete_bge_vectors(self, sub_id: int, doc_type: str = None) -> bool:
        """BGE 벡터 삭제 (구독키별 또는 특정 문서타입)"""
        try:
            with self.db_manager.get_connection() as connection:
                with connection.cursor() as cursor:
                    if doc_type:
                        index_name = self.get_bge_index_name(doc_type)
                        cursor.execute("""
                            DELETE FROM ai_faiss_indices
                            WHERE subscription_id = %s AND index_name = %s
                        """, (sub_id, index_name))
                        logger.info(f"🗑️ BGE 벡터 삭제: 구독키 {sub_id}, 타입 {doc_type}")
                    else:
                        cursor.execute("""
                            DELETE FROM ai_faiss_indices
                            WHERE subscription_id = %s AND index_name LIKE 'bge_%'
                        """, (sub_id,))
                        logger.info(f"🗑️ BGE 벡터 삭제: 구독키 {sub_id} 전체")

                    connection.commit()
                    return True

        except Exception as e:
            logger.error(f"❌ BGE 벡터 삭제 실패: {e}")
            return False

    def get_bge_stats(self) -> Dict[str, int]:
        """BGE 벡터 통계 조회"""
        try:
            with self.db_manager.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.execute("""
                        SELECT
                            subscription_id,
                            index_name,
                            total_docs
                        FROM ai_faiss_indices
                        WHERE index_name LIKE 'bge_%'
                        ORDER BY subscription_id, index_name
                    """)

                    rows = cursor.fetchall()
                    stats = {}
                    total = 0

                    for row in rows:
                        sub_id = row[0]  # subscription_id
                        index_name = row[1]  # index_name
                        doc_type = index_name.replace('bge_', '')
                        count = row[2] or 0  # total_docs

                        if sub_id not in stats:
                            stats[sub_id] = {}
                        stats[sub_id][doc_type] = count
                        total += count

                    stats['total'] = total
                    return stats

        except Exception as e:
            logger.error(f"❌ BGE 통계 조회 실패: {e}")
            return {}