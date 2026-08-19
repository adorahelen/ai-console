import multiprocessing
import hashlib
from datetime import datetime
import pickle
import asyncio
import base64
from typing import List, Dict, Tuple, Optional, Any, Union
from contextlib import contextmanager
import socket

import logging
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import os
import json
import numpy as np
import requests
import math
import time
import glob
import tiktoken
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from config_utils import ConfigManager
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial
import torch
import random
import yaml
from joblib import Parallel, delayed

from aibot_knowledge_graph import KnowledgeGraphGenerator, create_knowledge_graph

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

@dataclass
class Embedding:
    token_count: int
    vector: List[float]


class OfflineTiktoken:
    _cache_dir = Path(__file__).parent / "tiktoken_cache"
    _loaded_encodings = {}

    @classmethod
    def set_cache_dir(cls, cache_dir: str):
        cls._cache_dir = Path(cache_dir)

    @classmethod
    def get_encoding(cls, encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
        if encoding_name in cls._loaded_encodings:
            return cls._loaded_encodings[encoding_name]

        cache_file = cls._cache_dir / f"{encoding_name}.pkl"

        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    cache_data = pickle.load(f)

                encoding = tiktoken.Encoding(
                    name=cache_data["name"],
                    pat_str=cache_data["_pat_str"],
                    mergeable_ranks=cache_data["_mergeable_ranks"],
                    special_tokens=cache_data["_special_tokens"]
                )

                cls._loaded_encodings[encoding_name] = encoding
                return encoding

            except Exception as e:
                print(f"Warning: Failed to load cached encoding {encoding_name}: {e}")
                print("Falling back to online tiktoken...")

        try:
            encoding = tiktoken.get_encoding(encoding_name)
            cls._loaded_encodings[encoding_name] = encoding
            return encoding
        except Exception as e:
            raise RuntimeError(
                f"Failed to load encoding {encoding_name}. "
                f"Cache file not found at {cache_file} and online access failed. "
                f"Please generate cache files using generate_tiktoken_cache.py"
            ) from e

def is_internet_available(host="8.8.8.8", port=53, timeout=0.1):
    """
    빠른 인터넷 연결 확인 (Google DNS 서버 사용)
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except (socket.error, socket.timeout):
        return False

def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """
    인터넷 연결 상태에 따라 적절한 tiktoken 인코딩을 반환
    """
    if is_internet_available():
        try:
            # 인터넷이 연결되어 있으면 온라인 tiktoken 사용
            encoding = tiktoken.get_encoding(encoding_name)
            OfflineTiktoken._loaded_encodings[encoding_name] = encoding
            return encoding
        except Exception as e:
            # 인터넷은 연결되어 있지만 tiktoken 서버에 문제가 있는 경우
            print(f"Warning: Online tiktoken failed: {e}. Using offline mode...")
            return OfflineTiktoken.get_encoding(encoding_name)
    else:
        # 인터넷 연결이 없으면 바로 오프라인 모드 사용
        return OfflineTiktoken.get_encoding(encoding_name)


@dataclass
class FileMetadata:
    path: str
    hash: str
    size: int
    modified_time: float
    created_time: float


class EmbeddingModelAdapter:

    def __init__(self, model_type: str = "openai", local_model_path: Optional[str] = None):
        self.model_type = model_type
        self.local_model_path = local_model_path
        self._local_model = None
        self._bge_adapter = None

        # BGE 모드 체크
        from config_utils import ConfigManager
        config = ConfigManager()
        self.use_bge = config.config.get('embedding', 'use_bge_mode', fallback='False')
        if model_type == "local" and self.use_bge == 'True':
            # BGE 모드 사용
            self._init_bge_model()
            self.model_type = "bge_m3"
        elif model_type == "local":
            self._init_local_model()

    def _init_bge_model(self):
        """BGE-M3 모델 초기화"""
        if self._bge_adapter is not None:
            return

        try:
            from aibot_embedding_BGE import BGEEmbeddingAdapter

            # BGE 모델 경로
            bge_model_path = self.local_model_path or "/data/models/bge-m3"

            print(f"🚀 BGE-M3 모델 초기화: {bge_model_path}")
            self._bge_adapter = BGEEmbeddingAdapter(model_path=bge_model_path)
            print(f"✅ BGE-M3 모델 로드 완료")

        except Exception as e:
            print(f"❌ BGE-M3 모델 초기화 실패: {e}")
            raise

    def _init_local_model(self):
        if self._local_model is not None:
            return
            
        if not os.path.exists(self.local_model_path):
            raise FileNotFoundError(f"모델 경로를 찾을 수 없습니다: {self.local_model_path}")
        
        in_worker = multiprocessing.current_process().name != 'MainProcess'
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎯 GPU 메모리: {gpu_memory:.1f}GB")
            
            max_workers_for_gpu = 2
            
            if gpu_memory >= 8.0 and (not in_worker or multiprocessing.cpu_count() <= max_workers_for_gpu):
                device = 'cuda'
                torch.cuda.empty_cache()
                print(f"✅ GPU 모드 사용: CUDA")
            else:
                device = 'cpu'
                print(f"⚠️ CPU 모드 사용: {'워커 프로세스' if in_worker else 'GPU 메모리 부족'}")
        else:
            device = 'cpu'
            print("⚠️ CUDA 미지원 - CPU 모드 사용")
        
        from sentence_transformers import SentenceTransformer
        
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        safetensors_path = Path(self.local_model_path) / "model.safetensors"
        
        if safetensors_path.exists():
            print(f"✅ safetensors 형식 모델 감지: {safetensors_path}")
            self._local_model = SentenceTransformer(
                self.local_model_path, 
                device=device,
                trust_remote_code=False
            )
        else:
            
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*torch.load.*")
                
                try:
                    os.environ['TRANSFORMERS_OFFLINE'] = '1'
                    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
                    
                    self._local_model = SentenceTransformer(
                        self.local_model_path,
                        device=device,
                        trust_remote_code=False
                    )
                except Exception as inner_e:
                    print(f"⚠️ 첫 번째 로드 시도 실패: {inner_e}")
        
        print(f"✅ 로컬 임베딩 모델 로드 완료: {self.local_model_path} (디바이스: {device})")

    def _cleanup_gpu_memory(self):
        """GPU 메모리 강제 정리"""
        if torch.cuda.is_available():
            try:
                # 모든 GPU 스트림 동기화
                torch.cuda.synchronize()
                # GPU 캐시 정리
                torch.cuda.empty_cache()
                # 가비지 컬렉션 트리거
                import gc
                gc.collect()
            except Exception as e:
                print(f"⚠️ GPU 메모리 정리 실패: {e}")

    def _cleanup_gpu_memory_aggressive(self):
        """강력한 GPU 메모리 정리 (모델 재로드 포함)"""
        if torch.cuda.is_available():
            try:
                # 모델을 CPU로 임시 이동
                if hasattr(self, '_local_model') and self._local_model is not None:
                    device = self._local_model.device
                    if 'cuda' in str(device):
                        print("🔄 GPU 메모리 최적화를 위해 모델을 CPU로 임시 이동")
                        self._local_model = self._local_model.cpu()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # 모델을 다시 GPU로 이동
                        self._local_model = self._local_model.cuda()

                import gc
                gc.collect()
                print("✅ GPU 메모리 강력 정리 완료")
            except Exception as e:
                print(f"⚠️ 강력한 GPU 메모리 정리 실패: {e}")

    def generate_embedding_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        if self.model_type == "openai":
            return None

        if self.model_type == "bge_m3":
            # BGE 모드
            if self._bge_adapter is None:
                self._init_bge_model()
            return self._bge_adapter.generate_embedding_batch(texts)

        if self._local_model is None:
            self._init_local_model()

        try:
            with torch.no_grad():  # 메모리 최적화: 계산 그래프 추적 방지
                vectors = self._local_model.encode(
                    texts,
                    batch_size=min(32, len(texts)),
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True
                )

                # GPU 텐서인 경우 즉시 CPU로 이동
                if hasattr(vectors, 'cpu'):
                    cpu_vectors = vectors.cpu().numpy()
                    # 원본 GPU 텐서 해제
                    del vectors
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    vectors = cpu_vectors

            results = []
            for i, text in enumerate(texts):
                token_count = len(text.split())
                results.append({
                    "data": [{"embedding": vectors[i].tolist()}],
                    "usage": {"total_tokens": token_count},
                    "model": "local-embedding-model"
                })

            return results

        except Exception as e:
            print(f"⚠️ 배치 임베딩 생성 실패: {e}")
            return [self.generate_embedding(text) for text in texts]

        finally:
            # 최종 GPU 메모리 정리
            self._cleanup_gpu_memory()

    def generate_embedding(self, text: str) -> Dict[str, Any]:
        if self.model_type == "openai":
            return None

        if self.model_type == "bge_m3":
            # BGE 모드
            if self._bge_adapter is None:
                self._init_bge_model()
            return self._bge_adapter.generate_embedding(text)

        if self._local_model is None:
            self._init_local_model()

        try:
            with torch.no_grad():  # 메모리 최적화: 계산 그래프 추적 방지
                vector = self._local_model.encode(text)

                # GPU 텐서인 경우 즉시 CPU로 이동
                if hasattr(vector, 'cpu'):
                    vector = vector.cpu().numpy()

            normalized_vector = vector / np.linalg.norm(vector)
            token_count = len(text.split())

            return {
                "data": [{"embedding": normalized_vector.tolist()}],
                "usage": {"total_tokens": token_count},
                "model": "local-embedding-model"
            }

        finally:
            # GPU 메모리 정리
            self._cleanup_gpu_memory()


class EmbeddingGenerator:
    def __init__(self, config_path: str = "config.ini", save_db: bool = False, 
                 use_local_model: bool = False, local_model_path: Optional[str] = None,
                 batch_db_operations: bool = True):
        self.config_manager = ConfigManager(config_path)
        self.use_local_model = use_local_model
        self.batch_db_operations = batch_db_operations
        
        openai_config = self.config_manager.get_openai_config()
        self.api_key = openai_config.get('api_key') or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key and not use_local_model:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다. 'config.ini' 또는 환경 변수를 확인하세요.")
        
        paths_config = self.config_manager.get_paths_config()
        base_embedding_dir = Path(paths_config.get('aibot_embedding_dir', 'embeddings')).resolve()
        self.docs_dir = Path(paths_config.get('aibot_docs_dir', 'knowledge_base')).resolve()

        if use_local_model:
            self.embedding_dir = base_embedding_dir
            self.model_suffix = ""
        else:
            qa_embeddings_dir = base_embedding_dir.parent
            qa_embeddings_gpt_dir = qa_embeddings_dir.parent / (qa_embeddings_dir.name + "_gpt")
            self.embedding_dir = qa_embeddings_gpt_dir / base_embedding_dir.name
            self.model_suffix = "_gpt"
        
        self.model = "text-embedding-3-small"
        self.file_extension = "yaml"

        # BGE 모드 감지 및 지식그래프 비활성화
        model_config = self.config_manager.get_model_config()
        model_type = model_config.get('model', '').lower()
        use_bge = self.config_manager.config.get('embedding', 'use_bge_mode', fallback='False')
        is_bge_mode = ('gpt' in model_type and 'oss' in model_type and use_bge == 'True')

        if is_bge_mode:
            print("🚀 BGE 모드 감지: 지식그래프와 FAISS 인덱스가 비활성화됩니다")
            self.create_graph = False
            self.build_faiss = False  # BGE 모드에서는 FAISS 사용 안함
        else:
            self.create_graph = True

        self.workers = max(1, multiprocessing.cpu_count() - 1)
        
        if not use_local_model:
            self.api_workers = 16
            self.api_delay = 0.0
            self.batch_size = 256
        else:
            self.api_workers = 16
            self.api_delay = 0
            self.batch_size = 64

        self.use_ultra_fast_mode = use_local_model
        self.mega_batch_size = 128
        self.preload_all_texts = True

        if use_local_model:
            self.optimize_workers_for_gpu()

        self.graph_dir = self.embedding_dir.parent / "knowledge_graph"
        self.embedding_file = self.embedding_dir / f"embeddings_{self.file_extension}.data"
        self.metadata_file = self.embedding_dir / f"metadata_{self.file_extension}.json"
        
        self.embedding_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = get_encoding("cl100k_base")
        
        if self.create_graph:
            self.graph_dir.mkdir(parents=True, exist_ok=True)
            self.graph_generator = KnowledgeGraphGenerator(self.docs_dir, self.graph_dir)
        else:
            # BGE 모드에서는 지식그래프 관련 초기화 건너뜀
            self.graph_dir = None
            self.graph_generator = None
        
        if use_local_model:
            self.embedding_adapter = EmbeddingModelAdapter(
                model_type="local", 
                local_model_path=local_model_path
            )
            print(f"✅ 로컬 임베딩 모델 어댑터 초기화 완료")
        else:
            self.embedding_adapter = None
            print("✅ OpenAI 임베딩 API를 사용합니다.")

        self.save_db = save_db
        self.db_manager = None
        self.db_connection_pool = []

        # BGE 어댑터는 embedding_adapter에서 가져옴
        self.use_bge = self.config_manager.config.get('embedding', 'use_bge_mode', fallback='False')
        if use_local_model and self.use_bge == 'True' and self.embedding_adapter:
            self._bge_adapter = self.embedding_adapter._bge_adapter
            self.model_type = self.embedding_adapter.model_type
        else:
            self._bge_adapter = None

        # BGE DB 매니저 기본값 설정
        self._bge_db_manager = None
        
        if self.save_db:
            try:
                from aibot_db_manager import AibotDBManager
                from aibot_db_command import SQL_QUERIES
                
                self.db_manager = AibotDBManager(
                    config=self.config_manager,
                    query_properties=SQL_QUERIES
                )
                print("📊 DB 모드 활성화: MariaDB 사용")
                print(f"   - 배치 작업: {'활성화' if batch_db_operations else '비활성화'}")

                # BGE DB 매니저 초기화 (BGE 모드일 때)
                if use_local_model and self.use_bge == 'True':
                    try:
                        from aibot_embedding_BGE import BGEVectorDBManager
                        self._bge_db_manager = BGEVectorDBManager(self.config_manager, self.db_manager)
                        print(f"✅ BGE Vector DB 매니저 초기화 완료")
                    except Exception as e:
                        print(f"⚠️ BGE Vector DB 매니저 초기화 실패: {e}")
                        self._bge_db_manager = None
                else:
                    self._bge_db_manager = None
            except Exception as e:
                print(f"⚠️ DB 초기화 실패: {e}")
                print("파일 모드로 폴백합니다.")
                self.save_db = False
                self.db_manager = None

        self.build_faiss = FAISS_AVAILABLE
        self.faiss_general_path = self.embedding_dir / "faiss_general.index"
        self.faiss_qna_path = self.embedding_dir / "faiss_qna.index"
        self.faiss_metadata_path = self.embedding_dir / "faiss_metadata.json"
        
        print(f"\n📋 임베딩 설정:")
        print(f"- 문서 디렉토리: {self.docs_dir}")
        print(f"- 임베딩 디렉토리: {self.embedding_dir}")
        print(f"- 파일 확장자: {self.file_extension}")
        print(f"- 모델 타입: {'로컬 모델' if use_local_model else 'OpenAI API'}")
        print(f"- 임베딩 모델: {self.model if not use_local_model else '로컬 모델'}")
        print(f"- 저장 모드: {'MariaDB' if self.save_db else '파일'}")
        print(f"- 워커 수: {self.workers}")
        print(f"- API 워커 수: {self.api_workers}")
        print(f"- 배치 크기: {self.batch_size}")
        print(f"- 메가배치 크기: {self.mega_batch_size}")
        print(f"- FAISS 인덱스 구축: {'✅' if self.build_faiss else '❌'}")
        print(f"- 지식 그래프 생성: {'✅' if self.create_graph else '❌'}")
        print(f"- 🚀 극한 최적화: {'✅' if self.use_ultra_fast_mode else '❌'}")

    def optimize_workers_for_gpu(self):
        if torch.cuda.is_available() and self.use_local_model:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory >= 24:
                self.workers = 1
                self.batch_size = 128
                self.mega_batch_size = 256
                print(f"🎯 대용량 GPU 감지 ({gpu_memory:.1f}GB) - 극한 최적화 모드")
            elif gpu_memory >= 12:
                self.workers = 1
                self.batch_size = 64
                self.mega_batch_size = 128
                print(f"🎯 중급 GPU 감지 ({gpu_memory:.1f}GB) - 고속 모드")
            elif gpu_memory >= 8:
                self.workers = 1
                self.batch_size = 32
                self.mega_batch_size = 64
                print(f"🎯 표준 GPU 감지 ({gpu_memory:.1f}GB) - 최적화 모드")
            else:
                self.workers = max(1, multiprocessing.cpu_count() - 1)
                self.batch_size = 16
                self.mega_batch_size = 32
                print(f"🎯 저사양 GPU 감지 ({gpu_memory:.1f}GB) - 호환 모드")
            
            print(f"🎯 GPU 최적화 설정:")
            print(f"   - GPU 메모리: {gpu_memory:.1f}GB")
            print(f"   - 워커 수: {self.workers}")
            print(f"   - 배치 크기: {self.batch_size}")
            print(f"   - 메가 배치: {self.mega_batch_size}")
        else:
            print("⚠️ GPU 미감지 또는 OpenAI API 모드 - 기본 설정 유지")

    def preload_all_file_contents(self, file_paths: List[Path]) -> Dict[str, Tuple[str, str]]:
        print(f"📥 {len(file_paths)}개 파일 내용 메모리 로딩 중...")
        
        file_data = {}
        errors = []
        
        for relative_path in tqdm(file_paths, desc="파일 로딩", leave=False):
            try:
                file_path = f"{self.docs_dir}/{relative_path}"
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
                
                if self.file_extension in ['yaml', 'yml']:
                    embedding_text = self._extract_embedding_text_from_yaml(raw_content, relative_path)
                else:
                    embedding_text = raw_content
                
                file_data[relative_path] = (embedding_text, raw_content)
                
            except Exception as e:
                errors.append((relative_path, str(e)))
        
        if errors:
            print(f"⚠️ 파일 로딩 오류 {len(errors)}개")
            for path, error in errors[:5]:
                print(f"   - {path}: {error}")
        
        print(f"✅ {len(file_data)}개 파일 메모리 로딩 완료")
        return file_data

    def _extract_embedding_text_from_yaml(self, content: str, file_key: str) -> str:
        try:
            yaml_data = yaml.safe_load(content)
            if not isinstance(yaml_data, dict):
                return content
            
            embedding_parts = []
            
            for field, prefix in [
                ('question', '질문'),
                ('cot', '추론과정'),
                ('spec', '명세'),
                ('answer', '답변'),
                ('app_code', '앱코드'),
                ('require', '필수파라미터'),
                #('aliases', '관련키워드')
            ]:
                value = yaml_data.get(field, '')
                if value:
                    if field == 'require' and isinstance(value, list):
                        require_text = []
                        for req in value:
                            if isinstance(req, dict):
                                name = req.get('name', '')
                                type_info = req.get('type', '')
                                desc = req.get('description', '')
                                is_optional = req.get('optional', False)
                                default_val = req.get('default', '')
                                example_val = req.get('example', '')
                                param_info = f"{name}({type_info})"
                                if is_optional:
                                    param_info += "[선택사항]"
                                if default_val:
                                    param_info += f"[기본값:{default_val}]"
                                param_info += f": {desc}"
                                if example_val and not isinstance(example_val, list):
                                    param_info += f" 예시:{example_val}"
                                require_text.append(param_info)
                        value = ' '.join(require_text)
                    elif isinstance(value, list):
                        value = ' '.join(str(item) for item in value)
                    embedding_parts.append(f"{prefix}: {str(value).strip()}")
            
            for field in ['keywords', 'category', 'tags', 'description']:
                if field in yaml_data and yaml_data[field]:
                    value = yaml_data[field]
                    if isinstance(value, list):
                        value = ' '.join(str(item) for item in value)
                    embedding_parts.append(f"{field}: {str(value)}")
            
            embedding_text = ' '.join(embedding_parts) if embedding_parts else content
            
            if len(embedding_text) > 6000:
                embedding_text = embedding_text[:6000]
                
            return embedding_text
            
        except yaml.YAMLError:
            return content

    def _extract_question_from_yaml(self, content: str) -> str:
        """YAML 파일에서 Question 필드만 추출하여 반환"""
        try:
            yaml_data = yaml.safe_load(content)
            if not isinstance(yaml_data, dict):
                return ""

            question = yaml_data.get('question', '')
            if not question:
                return ""

            # Question 필드가 리스트인 경우 처리
            if isinstance(question, list):
                question = ' '.join(str(item) for item in question)

            # 정리 및 정규화
            question = str(question).strip()

            # 너무 짧은 question은 빈 문자열 반환
            if len(question) < 5:
                return ""

            return question

        except yaml.YAMLError:
            return ""

    def process_mega_batch(self, file_data: Dict[str, Tuple[str, str]]) -> Dict[str, Embedding]:
        if not self.use_local_model or not self.embedding_adapter:
            return {}
        
        print(f"🚀 메가 배치 처리 시작: {len(file_data)}개 파일")
        
        file_keys = list(file_data.keys())
        texts = [file_data[key][0] for key in file_keys]
        
        embeddings = {}
        batch_size = self.mega_batch_size
        
        start_time = time.time()
        
        with tqdm(total=len(texts), desc="🚀 메가배치 임베딩", ncols=100) as pbar:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_keys = file_keys[i:i + batch_size]
                
                try:
                    if self.embedding_adapter._local_model is None:
                        self.embedding_adapter._init_local_model()
                    
                    vectors = self.embedding_adapter._local_model.encode(
                        batch_texts,
                        batch_size=min(32, len(batch_texts)),
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                        device=self.embedding_adapter._local_model.device
                    )
                    
                    if hasattr(vectors, 'cpu'):
                        vectors = vectors.cpu().numpy()
                    
                    for j, (key, text) in enumerate(zip(batch_keys, batch_texts)):
                        token_count = len(text.split())

                        # Question 임베딩 제거됨

                        embedding = Embedding(
                            token_count=token_count,
                            vector=vectors[j].tolist()
                        )
                        embeddings[key] = embedding
                    
                    pbar.update(len(batch_texts))
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"⚠️ 배치 처리 오류: {e}")
                    for key, text in zip(batch_keys, batch_texts):
                        try:
                            result = self.embedding_adapter.generate_embedding(text)
                            token_count = result["usage"]["total_tokens"]
                            vector = result["data"][0]["embedding"]
                            embeddings[key] = Embedding(token_count=token_count, vector=vector)
                        except:
                            pass
                    pbar.update(len(batch_texts))
        
        elapsed_time = time.time() - start_time
        speed = len(embeddings) / elapsed_time if elapsed_time > 0 else 0
        
        print(f"✅ 메가 배치 처리 완료:")
        print(f"   - 처리된 파일: {len(embeddings)}개")
        print(f"   - 소요 시간: {elapsed_time:.2f}초")
        print(f"   - 🚀 처리 속도: {speed:.1f} 파일/초")
        
        return embeddings

    def process_files_batch(self, file_infos: List[Tuple[Path, str]]) -> List[Tuple[str, Optional[Embedding], Optional[Exception], Optional[str]]]:
        if not self.use_local_model or not self.embedding_adapter:
            return []
        
        try:
            texts = []
            file_keys = []
            contents = []
            
            for file_path, file_key in file_infos:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    
                    if self.file_extension in ['yaml', 'yml']:
                        embedding_text = self._extract_embedding_text_from_yaml(file_content, file_key)
                    else:
                        embedding_text = file_content
                    
                    texts.append(embedding_text)
                    file_keys.append(file_key)
                    contents.append(file_content if self.save_db else None)
                    
                except Exception as e:
                    print(f"⚠️ 파일 읽기 오류 ({file_key}): {e}")
                    continue
            
            if not texts:
                return []
            
            results = self.embedding_adapter.generate_embedding_batch(texts)
            
            batch_results = []
            for i, (file_key, content) in enumerate(zip(file_keys, contents)):
                if i < len(results) and results[i]:
                    result = results[i]
                    token_count = result["usage"]["total_tokens"]
                    vector = result["data"][0]["embedding"]

                    # Question 임베딩 제거됨

                    embedding = Embedding(
                        token_count=token_count,
                        vector=vector
                    )
                    batch_results.append((file_key, embedding, None, content))
                else:
                    batch_results.append((file_key, None, Exception("배치 처리 실패"), None))
            
            return batch_results
            
        except Exception as e:
            print(f"⚠️ 배치 처리 전체 실패: {e}")
            return []

    def process_file(
        self,
        file_info: Tuple[Union[Path, str], str],
        api_key: str,
        delay: float = 0.001
    ) -> Tuple[str, Optional[Embedding], Optional[Exception], Optional[str]]:
        raw_input, file_key = file_info

        if self.use_local_model:
            max_retries = 2
            base_delay = 0.1
        else:
            max_retries = 3
            base_delay = 0.2

        content: Optional[str] = None

        for attempt in range(max_retries + 1):
            try:
                if isinstance(raw_input, (Path, )):
                    with open(raw_input, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                elif isinstance(raw_input, str):
                    file_content = raw_input
                else:
                    return file_key, None, TypeError(f"file_info[0] must be Path or str, got {type(raw_input)}"), None

                if self.save_db:
                    content = file_content

                if self.file_extension in ['yaml', 'yml']:
                    embedding_text = self._extract_embedding_text_from_yaml(file_content, file_key)
                    # Question 텍스트 추출
                    question_text = self._extract_question_from_yaml(file_content)
                else:
                    embedding_text = file_content
                    question_text = ""

                if self.use_local_model:
                    # 전체 텍스트 임베딩
                    result = self.embedding_adapter.generate_embedding(embedding_text)
                    token_count = result["usage"]["total_tokens"]
                    vector = result["data"][0]["embedding"]

                    # Question 임베딩 제거됨
                    print(f"   ℹ️ Question 텍스트 처리 건너뜀")

                    embedding = Embedding(
                        token_count=token_count,
                        vector=vector
                    )
                    return file_key, embedding, None, content

                else:
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "User-Agent": "EmbeddingGenerator/2.0"
                    }
                    data = {
                        "input": embedding_text,
                        "model": self.model,
                        "encoding_format": "float"
                    }

                    response = requests.post(
                        "https://api.openai.com/v1/embeddings",
                        headers=headers,
                        json=data,
                        timeout=(5, 15),
                        stream=False
                    )

                    if response.status_code == 200:
                        result = response.json()
                        token_count = result["usage"]["total_tokens"]
                        vector = result["data"][0]["embedding"]

                        # Question 임베딩 제거됨

                        embedding = Embedding(
                            token_count=token_count,
                            vector=vector
                        )

                        if delay > 0:
                            time.sleep(delay)

                        return file_key, embedding, None, content

                    elif response.status_code == 429:
                        wait_time = min(5.0, base_delay * (1.5 ** attempt) + random.uniform(0, 0.5))
                        if attempt < max_retries:
                            time.sleep(wait_time)
                            continue
                        else:
                            return file_key, None, Exception(f"Rate limit: {file_key}"), None

                    elif response.status_code in [500, 502, 503, 504]:
                        wait_time = min(3.0, base_delay * (1.2 ** attempt))
                        if attempt < max_retries:
                            time.sleep(wait_time)
                            continue
                        else:
                            return file_key, None, Exception(f"서버 에러 ({response.status_code}): {file_key}"), None

                    else:
                        return file_key, None, Exception(f"API 에러 ({response.status_code}): {response.text[:200]}"), None

            except requests.exceptions.Timeout:
                wait_time = min(2.0, base_delay * (1.2 ** attempt))
                if attempt < max_retries:
                    time.sleep(wait_time)
                    continue
                else:
                    return file_key, None, Exception(f"타임아웃: {file_key}"), None

            except UnicodeDecodeError as e:
                return file_key, None, Exception(f"인코딩 에러: {e}"), None

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(0.5)
                    continue
                else:
                    return file_key, None, e, None

        return file_key, None, Exception(f"모든 재시도 실패: {file_key}"), None
    
    def calculate_file_hash(self, file_path: Path) -> str:
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return ""
    
    def get_db_metadatas(self, results):
        paths = set()
        datas = {}
        
        for checksum in results:
            source = checksum['source'].split('/', 1)[-1] if checksum['source'].startswith(('llama/', 'gpt/')) else checksum['source']
            paths.add(source)
            datas[source] = FileMetadata(
                path=str(source),
                hash=checksum['vector_hash'],
                size=checksum['token_count'],
                modified_time=checksum['updated_ts'],
                created_time=checksum['created_ts']
            )
        
        return paths, datas

    def get_file_metadata(self, file_path: Path) -> FileMetadata:
        stat = file_path.stat()
        relative_path = file_path.relative_to(self.docs_dir)
        
        return FileMetadata(
            path=str(relative_path),
            hash=self.calculate_file_hash(file_path),
            size=stat.st_size,
            modified_time=stat.st_mtime,
            created_time=stat.st_ctime
        )

    def load_metadata(self, sub_id, model) -> Tuple[Dict[str, FileMetadata]]:
        if self.save_db and self.db_manager:
            print(f"DB 메타데이터 로드")
            try:
                docs = self.db_manager.get_prompts(s_id=sub_id)
                metadata = {}
                for doc in docs:
                    source = doc['source'].split('/', 1)[-1] if doc['source'].startswith(('llama/', 'gpt/')) else doc['source']
                    file_path = self.docs_dir / source
                    if file_path.exists():
                        metadata[source] = self.get_file_metadata(file_path)
                
                return metadata
            except Exception as e:
                print(f"⚠️ DB 메타데이터 로드 오류: {e}")
                return {}
        else:
            if not self.metadata_file.exists():
                return {}
            
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = {}
                for path, meta_dict in data.items():
                    metadata[path] = FileMetadata(
                        path=meta_dict['path'],
                        hash=meta_dict['hash'],
                        size=meta_dict['size'],
                        modified_time=meta_dict['modified_time'],
                        created_time=meta_dict['created_time']
                    )
                
                return metadata
            except Exception as e:
                print(f"⚠️ 메타데이터 로드 오류: {e}")
                return {}
            
    def save_db_metadata(self, sub_id, model):
        try:
            print(f"메타데이터 저장 시작")
            results = self.db_manager.get_vector_checksums(sub_id)
            
            file_path = self.embedding_dir / f"metadata_{model}_{sub_id}.json"
            
            data = {}
            
            for checksum in results:
                source = checksum['source'].split('/', 1)[-1] if checksum['source'].startswith(('llama/', 'gpt/')) else checksum['source']
                data[source] = {
                    'path': str(source),
                    'hash': checksum['vector_hash'],
                    'size': checksum['token_count'],
                    'modified_time': checksum['updated_ts'],
                    'created_time': checksum['created_ts']
                }
            
            temp_file = str(file_path) + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if file_path.exists():
                os.remove(file_path)
            
            os.rename(temp_file, file_path)
            
        except Exception as e:
            print(f"⚠️ DB 메타데이터 저장 실패: {e}")

    def save_metadata(self, metadata: Dict[str, FileMetadata]) -> None:
        if self.save_db and self.db_manager:
            return
        
        try:
            data = {}
            for path, meta in metadata.items():
                data[path] = {
                    'path': meta.path,
                    'hash': meta.hash,
                    'size': meta.size,
                    'modified_time': meta.modified_time,
                    'created_time': meta.created_time
                }
            
            temp_file = str(self.metadata_file) + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            if self.metadata_file.exists():
                os.remove(self.metadata_file)
            
            os.rename(temp_file, self.metadata_file)
            
        except Exception as e:
            print(f"⚠️ 메타데이터 저장 오류: {e}")

    def analyze_file_changes(self, sub_id, model) -> Tuple[List[Path], List[str], List[Path], Dict[str, FileMetadata]]:
        old_metadata = self.load_metadata(sub_id, model)
    
        if self.save_db and self.db_manager:
            is_initial_run = len(old_metadata) == 0
        else:
            is_initial_run = not self.embedding_file.exists() or not self.metadata_file.exists()
        
        if self.file_extension == 'yaml':
            file_patterns = ['*.yaml', '*.yml']
            current_files = []
            for pattern in file_patterns:
                current_files.extend(list(self.docs_dir.rglob(pattern)))
        else:
            current_files = list(self.docs_dir.rglob(f"*.{self.file_extension}"))
        
        print(f"🔍 파일 분석:")
        print(f"   - 현재 파일 수: {len(current_files)}개")
        print(f"   - 기존 메타데이터 수: {len(old_metadata)}개")
        print(f"   - 초기 실행: {'예' if is_initial_run else '아니오'}")
        
        current_paths = set()
        current_metadata = {}
        
        for file_path in current_files:
            relative_path = str(file_path.relative_to(self.docs_dir))
            current_paths.add(relative_path)
            current_metadata[relative_path] = self.get_file_metadata(file_path)
        
        old_paths = set(old_metadata.keys())
        
        new_files = []
        deleted_files = []
        modified_files = []
        
        if is_initial_run:
            print("🆕 초기 실행 감지: 모든 파일 처리")
            for file_path in current_files:
                new_files.append(file_path)
        else:
            print("🔄 증분 업데이트 모드")
            
            existing_embeddings, prompts = self.load_embeddings_from_file(sub_id, model)
            print(f"   - 기존 임베딩 수: {len(existing_embeddings)}개")
            
            for path in current_paths - old_paths:
                full_path = self.docs_dir / path
                new_files.append(full_path)
                print(f"   🆕 새 파일: {path}")
            
            for path in current_paths & old_paths:
                if path not in existing_embeddings:
                    full_path = self.docs_dir / path
                    new_files.append(full_path)
                    print(f"   🔄 임베딩 누락 파일: {path}")
            
            deleted_files = list(old_paths - current_paths)
            for path in deleted_files:
                print(f"   🗑️ 삭제된 파일: {path}")
            
            for path in current_paths & old_paths:
                if path in existing_embeddings:
                    current_meta = current_metadata[path]
                    old_meta = old_metadata[path]
                    
                    if current_meta.hash != old_meta.hash:
                        full_path = self.docs_dir / path
                        modified_files.append(full_path)
                        print(f"   ✏️ 수정된 파일: {path}")
        
        print(f"📊 최종 분석 결과:")
        print(f"   🆕 새로운 파일: {len(new_files)}개")
        print(f"   🗑️ 삭제된 파일: {len(deleted_files)}개") 
        print(f"   ✏️ 수정된 파일: {len(modified_files)}개")
        
        return new_files, deleted_files, modified_files, current_metadata
    
    def analyze_db_changes(self, sub_id, model) -> Tuple[List[Path], List[str], List[Path], Dict[str, FileMetadata]]:
        try:
            file_path = self.embedding_dir / f"metadata_{model}_{sub_id}.json"
            print(f"DB 메타데이터 로드: {file_path}")
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                old_metadata = {}
                for path, meta_dict in data.items():
                    old_metadata[path] = FileMetadata(
                        path=meta_dict['path'],
                        hash=meta_dict['hash'],
                        size=meta_dict['size'],
                        modified_time=meta_dict['modified_time'],
                        created_time=meta_dict['created_time']
                    )
            else:
                old_metadata = {}

        except Exception as e:
            print(f"⚠️ 메타데이터 로드 오류: {e}")
            old_metadata = {}

        is_initial_run = False
        '''if self.save_db and self.db_manager:
            is_initial_run = len(old_metadata) == 0
        else:
            is_initial_run = not self.embedding_file.exists() or not self.metadata_file.exists()'''
        
        current_files = []
        
        print(f"🔍 파일 분석:")
        print(f"   - 현재 파일 수: {len(current_files)}개")
        print(f"   - 기존 메타데이터 수: {len(old_metadata)}개")
        print(f"   - 초기 실행: {'예' if is_initial_run else '아니오'}")
        
        current_paths = set()
        current_metadata = {}
        
        results = self.db_manager.get_vector_checksums(sub_id)

        current_paths, current_metadata = self.get_db_metadatas(results)

        old_paths = set(old_metadata.keys())

        new_files = []
        deleted_files = []
        modified_files = []
        
        if is_initial_run:
            print("🆕 초기 실행 감지: 모든 파일 처리")
            for file_path in current_paths:
                new_files.append(file_path)
        else:
            print("🔄 증분 업데이트 모드")
            
            existing_embeddings, prompts = self.load_embeddings_from_file(sub_id, model)
            print(f"   - 기존 임베딩 수: {len(existing_embeddings)}개")
            
            for path in current_paths - old_paths:
                full_path = self.docs_dir / path
                new_files.append(path)
                print(f"   🆕 새 파일: {path}")
            
            for path in current_paths & old_paths:
                if path not in existing_embeddings:
                    full_path = self.docs_dir / path
                    new_files.append(path)
                    print(f"   🔄 임베딩 누락 파일: {path}")
            
            deleted_files = list(old_paths - current_paths)
            for path in deleted_files:
                print(f"   🗑️ 삭제된 파일: {path}")
            
            for path in current_paths & old_paths:
                if path in existing_embeddings:
                    current_meta = current_metadata[path]
                    old_meta = old_metadata[path]
                    
                    if current_meta.hash != old_meta.hash:
                        full_path = self.docs_dir / path
                        modified_files.append(path)
                        print(f"   ✏️ 수정된 파일: {path}")
        
        print(f"📊 최종 분석 결과:")
        print(f"   🆕 새로운 파일: {len(new_files)}개")
        print(f"   🗑️ 삭제된 파일: {len(deleted_files)}개") 
        print(f"   ✏️ 수정된 파일: {len(modified_files)}개")
        
        return new_files, deleted_files, modified_files, current_metadata

    def remove_deleted_embeddings(self, sub_id, deleted_files: List[str], embeddings: Dict[str, Embedding]) -> Dict[str, Embedding]:
        removed_count = 0
        db_deleted_count = 0

        for file_path in deleted_files:
            if file_path in embeddings:
                del embeddings[file_path]
                removed_count += 1

            if self.save_db and self.db_manager:
                try:
                    name = file_path.split("/")[-1].replace(".yaml", "")
                    data = self.db_manager.get_prompt_by_name(sub_id=sub_id, name=name)
                    if data:
                        self.db_manager.delete_prompt(sub_id=sub_id, guid=data['guid'])
                        db_deleted_count += 1
                        print(f"   ✅ DB에서 삭제: {file_path}")
                except Exception as e:
                    print(f"   ❌ DB에서 문서 삭제 실패 ({file_path}): {e}")

        if removed_count > 0:
            print(f"🗑️ 메모리에서 삭제된 임베딩: {removed_count}개")
        if db_deleted_count > 0:
            print(f"🗑️ DB에서 삭제된 문서: {db_deleted_count}개")

        return embeddings

    def write_embeddings_to_file(self, sub_id, embeddings: Dict[str, Embedding], file_contents: Dict[str, str] = None, tags: str = None) -> None:

        # BGE 모드 디버그 정보
        model_type_val = getattr(self, 'model_type', '없음')
        print(f"🔍 BGE 모드 체크: model_type={model_type_val}, use_bge={self.use_bge}, _bge_adapter={'있음' if self._bge_adapter else '없음'}")

        # BGE 모드일 때는 Qdrant 서버로만 업로드 (MariaDB 건너뛰기)
        if hasattr(self, 'model_type') and self.model_type == "bge_m3" and self._bge_adapter:
            print(f"🚀 BGE 모드 - Qdrant 서버로 직접 업로드...")

            # BGE 임베딩 데이터 준비
            bge_embeddings_data = []
            for key, embedding in embeddings.items():
                if isinstance(embedding.vector, dict) and embedding.vector.get("type") == "bge_m3":
                    content = ""
                    if file_contents and key in file_contents:
                        content = file_contents[key]
                    else:
                        file_path = self.docs_dir / key
                        if file_path.exists():
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()

                    bge_embeddings_data.append({
                        'file_key': key,
                        'text': content,
                        'embedding': embedding.vector
                    })

            if bge_embeddings_data:
                if hasattr(self, '_bge_db_manager') and self._bge_db_manager:
                    # 대량 업로드 시 타임아웃 방지를 위해 청크 단위로 업로드
                    chunk_size = 20  # 한번에 20개씩 업로드 (멀티벡터 용량 고려)
                    total_success = 0
                    total_chunks = (len(bge_embeddings_data) + chunk_size - 1) // chunk_size

                    # 시작 ID를 컬렉션에서 한 번만 조회
                    start_id = self._bge_db_manager.get_collection_count()
                    print(f"📤 BGE 임베딩 {len(bge_embeddings_data)}개를 {total_chunks}개 청크로 나누어 업로드... (시작 ID: {start_id})")

                    for i in range(0, len(bge_embeddings_data), chunk_size):
                        chunk = bge_embeddings_data[i:i+chunk_size]
                        chunk_num = (i // chunk_size) + 1

                        print(f"📦 청크 {chunk_num}/{total_chunks}: {len(chunk)}개 업로드 중... (ID: {start_id}~)")
                        uploaded_count = self._bge_db_manager.store_bge_embeddings(sub_id, chunk, start_id)
                        if uploaded_count >= 0:
                            total_success += uploaded_count
                            start_id += uploaded_count  # 다음 청크의 시작 ID 갱신
                            print(f"✅ 청크 {chunk_num} 업로드 완료 ({uploaded_count}개)")
                        else:
                            print(f"❌ 청크 {chunk_num} 업로드 실패")

                    print(f"✅ BGE 임베딩 업로드 완료: {total_success}/{len(bge_embeddings_data)}개")
                else:
                    print(f"⚠️ BGE DB 매니저가 초기화되지 않음")
            return

        # 일반 모드일 때만 MariaDB 저장
        if self.save_db and self.db_manager:
            try:
                print(f"DB 저장 시작")
                if self.batch_db_operations:
                    self._write_embeddings_batch_db(sub_id, embeddings, file_contents, tags)
                else:
                    self._write_embeddings_individual_db(embeddings, file_contents, tags)

            except Exception as e:
                print(f"⚠️ DB 저장 실패: {e}")
                self._write_embeddings_to_file_fallback(embeddings)
        else:
            self._write_embeddings_to_file_fallback(embeddings)
    
    def _write_embeddings_batch_db(self, sub_id, embeddings: Dict[str, Embedding], file_contents: Dict[str, str] = None, tags: str = None) -> None:
        print(f"🚀 배치 DB 작업 시작... {len(embeddings)}개")
        if tags:
            print(f"   ✅ Tags 정보: {tags}")

        batch_data = []
        update_data = []

        for key, embedding in embeddings.items():
            content = ""
            content_yaml = {}
            
            if file_contents and key in file_contents:
                content = file_contents[key]
            else:
                file_path = self.docs_dir / key
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
            
            try:
                content_yaml = yaml.safe_load(content) if content else {}
            except:
                content_yaml = {}
                
            # BGE 벡터 체크 및 직렬화
            if isinstance(embedding.vector, dict) and embedding.vector.get("type") == "bge_m3":
                # BGE 멀티벡터를 pickle로 직렬화
                import pickle
                vector_bytes = pickle.dumps(embedding.vector)
            else:
                # 기존 방식 (numpy array to bytes)
                vector_array = self.to_vector_array(embedding.vector)
                vector_bytes = vector_array.astype(np.float32).tobytes()

            # Question 벡터 제거됨

            name = key.split("/")[-1].replace(".yaml", "")
            prompt_type = key.split("/")[0] if "/" in key else "general"

            guid = content_yaml.get('guid', None)

            # YAML 파일에서 tags 추출 (파라미터로 받은 tags가 없을 경우)
            file_tags = tags if tags != "None" else None  # 파라미터로 받은 tags 우선 사용 ("None" 문자열 체크)
            if not file_tags and content_yaml:
                file_tags = content_yaml.get('tags', None)
                # "None" 문자열을 실제 None으로 변환
                if file_tags == "None":
                    file_tags = None
                elif file_tags:
                    print(f"   📌 파일에서 tags 추출: {file_tags} ({key})")
            
            if guid:
                existing_data = self.db_manager.get_prompt_by_guid(sub_id=sub_id, guid=guid)
            else:
                existing_data = self.db_manager.get_prompt_by_name(sub_id=sub_id, name=name)
                
            model = "llama" if self.use_local_model else "gpt"
            if model == "gpt":
                source = f"{model}/{key}"
            else:
                source = key
             
            if existing_data:
                update_data.append({
                    'old': existing_data,
                    'guid': existing_data['guid'],
                    'name': name,
                    'description': content_yaml.get('description', ''),
                    'prompt_type': prompt_type,
                    'prompt_data': content,
                    'token_count': embedding.token_count,
                    'embedding_vector': vector_bytes,
                    'source': source,
                    'tags': file_tags
                })
            else:
                batch_data.append({
                    'name': name,
                    'sub_id': sub_id,
                    'description': content_yaml.get('description', ''),
                    'guid': content_yaml.get('guid', None),
                    'prompt_type': prompt_type,
                    'prompt_data': content,
                    'source': source,
                    'token_count': embedding.token_count,
                    'embedding_vector': vector_bytes,
                    'tags': file_tags
                })
        
        if batch_data:
            print(f"   📝 새 데이터 {len(batch_data)}개 일괄 삽입...")
            success = self.db_manager.batch_insert_prompts(batch_data)
            print(f"   ✅ 삽입 완료: {success}개")
        
        if update_data:
            print(f"   🔄 기존 데이터 {len(update_data)}개 일괄 업데이트...")
            success = self.db_manager.batch_update_prompts(update_data)
            print(f"   ✅ 업데이트 완료: {success}개")
        
        print(f"✅ 배치 DB 작업 완료: 총 {len(embeddings)}개 처리")
        
    def to_vector_array(self, data, dtype=np.float32):
        if isinstance(data, np.ndarray):
            return data.astype(dtype)

        if isinstance(data, (list, tuple)):
            return np.array(data, dtype=dtype)

        if isinstance(data, (bytes, bytearray)):
            try:
                return np.frombuffer(data, dtype=dtype)
            except Exception:
                try:
                    obj = pickle.loads(data)
                    return np.array(obj, dtype=dtype)
                except Exception:
                    raise ValueError("bytes 데이터를 float32 array로 변환할 수 없습니다")

        if isinstance(data, str):
            try:
                obj = json.loads(data)
                return np.array(obj, dtype=dtype)
            except Exception:
                raise ValueError("string 데이터를 JSON으로 변환할 수 없습니다")

        raise TypeError(f"지원하지 않는 타입: {type(data)}")
    
    def _write_embeddings_individual_db(self, embeddings: Dict[str, Embedding], file_contents: Dict[str, str] = None, tags: str = None) -> None:
        success_count = 0
        error_count = 0
        
        for key, embedding in embeddings.items():
            try:
                content = ""
                if file_contents and key in file_contents:
                    content = file_contents[key]
                else:
                    file_path = self.docs_dir / key
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                
                content_yaml = yaml.safe_load(content) if content else {}

                # BGE 벡터 체크 및 직렬화
                if isinstance(embedding.vector, dict) and embedding.vector.get("type") == "bge_m3":
                    import pickle
                    vector_data = pickle.dumps(embedding.vector)
                else:
                    vector_array = np.array(embedding.vector, dtype=np.float32)
                    vector_data = vector_array

                name = key.split("/")[-1].replace(".yaml", "")
                prompt_type = key.split("/")[0] if "/" in key else "general"

                # YAML 파일에서 tags 추출 (파라미터로 받은 tags가 없을 경우)
                file_tags = tags if tags != "None" else None  # 파라미터로 받은 tags 우선 사용 ("None" 문자열 체크)
                if not file_tags and content_yaml:
                    file_tags = content_yaml.get('tags', None)
                    # "None" 문자열을 실제 None으로 변환
                    if file_tags == "None":
                        file_tags = None
                    elif file_tags:
                        print(f"   📌 파일에서 tags 추출: {file_tags} ({key})")
                
                sub_id = 0 if self.use_local_model else 1
                data = self.db_manager.get_prompt_by_name(sub_id=sub_id, name=name)
                
                if data:
                    self.db_manager.update_prompt_values(
                        old=data,
                        sub_id=data['subscription_id'],
                        guid=data['guid'],
                        name=name,
                        description=content_yaml.get('description', ''),
                        prompt_type=prompt_type,
                        prompt_data=content,
                        embedding_vector=vector_data,
                        source=key,
                        tags=file_tags
                    )
                else:
                    self.db_manager.insert_prompt_values(
                        name=name,
                        description=content_yaml.get('description', ''),
                        prompt_type=prompt_type,
                        prompt_data=content,
                        model="llama" if self.use_local_model else "gpt",
                        source=key,
                        embedding_vector=vector_data,
                        tags=file_tags
                    )
                
                success_count += 1
            
            except Exception as e:
                print(f"⚠️ 개별 임베딩 저장 실패 ({key}): {e}")
                error_count += 1
        
        print(f"DB에 임베딩 저장 완료: 성공 {success_count}개, 실패 {error_count}개")
    
    def _write_embeddings_to_file_fallback(self, embeddings: Dict[str, Embedding]) -> None:
        temp_file = str(self.embedding_file) + ".tmp"
        
        with open(temp_file, 'w', encoding='utf-8') as f:
            for key, embedding in embeddings.items():
                vector_str = ','.join([str(v) for v in embedding.vector])
                f.write(f"{key}:{embedding.token_count}:{vector_str}\n")
        
        if self.embedding_file.exists():
            os.remove(self.embedding_file)
        
        os.rename(temp_file, self.embedding_file)
        print(f"파일에 임베딩 {len(embeddings)}개 저장 완료")

    def decode_vector(self, v, source_info=None):
        if isinstance(v, str):
            try:
                v = base64.b64decode(v)
            except Exception as e:
                if source_info:
                    print(f"❌ base64 디코딩 실패한 파일: {source_info} - {e}")
                else:
                    print(f"base64 decoding failed: {e}")
                return None

        try:
            return np.frombuffer(v, dtype=np.float32)
        except Exception as e:
            '''if source_info:
                print(f"❌ 벡터 디코딩 실패한 파일: {source_info} - 벡터 데이터: {str(v)[:100]}...")
            else:
                print(f"Error 발생한 벡터 : {v}")'''
            return None
            
    def load_embeddings_from_file(self, sub_id, model):
        embeddings = {}
        prompts_docs = {}
        if self.save_db and self.db_manager:
            try:
                prompts = self.db_manager.get_prompts(s_id=sub_id, model=model)
                
                for prompt in prompts:
                    source = prompt.get('source', prompt['name'])
                    key = source.split('/', 1)[-1] if source.startswith(('llama/', 'gpt/')) else source

                    if prompt.get('vector'):
                        vector_data = self.decode_vector(prompt['vector'], source_info=f"{source} (ID: {prompt.get('id', 'Unknown')})")

                        if vector_data is None:
                            continue  # 디코딩 실패한 경우 건너뛰기

                        if isinstance(vector_data, np.ndarray):
                            vector = vector_data.tolist()
                        else:
                            vector = vector_data

                        embeddings[key] = Embedding(
                            token_count=len(vector),
                            vector=vector
                        )
                    if prompt.get('prompt'):
                        prompts_docs[key] = prompt.get('prompt')
                
                print(f"DB에서 임베딩 {len(embeddings)}개 로드 완료")
            except Exception as e:
                print(f"⚠️ DB 임베딩 로드 오류: {e}")
                if self.embedding_file.exists():
                    return self._load_embeddings_from_file_fallback()
        else:
            return self._load_embeddings_from_file_fallback()
        
        return embeddings, prompts_docs
    
    def _load_embeddings_from_file_fallback(self):
        embeddings = {}
        prompts = {}
        if not self.embedding_file.exists():
            return embeddings, prompts
        
        with open(self.embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                parts = line.strip().split(':', 2)
                if len(parts) == 3:
                    key, token_count_str, vector_str = parts
                    token_count = int(token_count_str)
                else:
                    key, vector_str = parts
                    token_count = 0
                
                vector = [float(x) for x in vector_str.split(',')]
                embeddings[key] = Embedding(token_count=token_count, vector=vector)
                prompts[key] = key
        
        print(f"파일에서 임베딩 {len(embeddings)}개 로드 완료")
        return embeddings, prompts

    def rebuild_faiss_if_needed(self, sub_id, embeddings: Dict[str, Embedding], all_files: List[Path], 
                           has_deletions: bool = False, has_modifications: bool = False) -> None:
        if not self.build_faiss:
            return
        
        status = self.check_faiss_index_status(sub_id)
        missing_indexes = not all(status.values())
        
        should_rebuild = (missing_indexes or has_deletions or has_modifications)
        
        if should_rebuild:
            print("🔄 의도별 FAISS 인덱스 재구축이 필요합니다...")
            self.build_faiss_indexes(sub_id, embeddings)
        else:
            print("✅ 모든 의도별 FAISS 인덱스가 최신 상태입니다.")
    
    def check_faiss_index_status(self, sub_id) -> Dict[str, bool]:
        intents = ['general', 'qna', 'action', 'plan'] 
        status = {}
        
        if self.save_db and self.db_manager:
            for intent in intents:
                index_name = f"faiss_{intent}"
                result = self.db_manager.load_faiss_index(sub_id, index_name)
                status[intent] = result is not None
        else:
            for intent in intents:
                index_path = self.embedding_dir / f"faiss_{intent}.index"
                metadata_path = self.embedding_dir / f"faiss_{intent}_metadata.json"
                status[intent] = index_path.exists() and metadata_path.exists()
        
        return status

    def build_faiss_indexes(self, sub_id, embeddings: Dict[str, Embedding]) -> None:
        if not embeddings:
            print("⚠️ 임베딩 데이터가 없어 FAISS 인덱스를 구축할 수 없습니다.")
            return

        print(f"📊 의도별 FAISS 인덱스 구축: {len(embeddings)}개 문서")

        dim_groups = {}
        embedding_dims = set()

        for key, emb in embeddings.items():
            dim_len = len(emb.vector)
            embedding_dims.add(dim_len)
            dim_groups.setdefault(dim_len, []).append(key)

        if len(embedding_dims) > 1:
            details = {dim: keys for dim, keys in dim_groups.items()}
            raise ValueError(f"임베딩 차원 불일치 발견! 상세 내역: {details}")
        
        embedding_dim = embedding_dims.pop()
        print(f"   - 임베딩 차원: {embedding_dim}")
        
        try:
            file_keys = list(embeddings.keys())
            all_vectors = self.to_vector_array([embeddings[key].vector for key in file_keys])
            faiss.normalize_L2(all_vectors)
            
            intent_file_mapping = {'general': [], 'qna': [], 'action': [], 'plan': []}
            
            for i, file_key in enumerate(file_keys):
                path_parts = file_key.replace('\\', '/').split('/')
                
                if len(path_parts) >= 2:
                    first_dir = path_parts[0].lower()
                    if first_dir in intent_file_mapping:
                        intent_file_mapping[first_dir].append((i, file_key))
                    else:
                        intent_file_mapping['general'].append((i, file_key))
                else:
                    intent_file_mapping['general'].append((i, file_key))
            
            for intent in ['general', 'qna', 'action', 'plan']:
                if intent == 'general':
                    vectors = all_vectors
                    keys = file_keys
                else:
                    file_indices_and_keys = intent_file_mapping[intent]
                    if not file_indices_and_keys:
                        continue
                    
                    indices = [idx for idx, _ in file_indices_and_keys]
                    keys = [key for _, key in file_indices_and_keys]
                    vectors = all_vectors[indices]
                
                index = faiss.IndexFlatIP(embedding_dim)
                index.add(vectors)
                
                metadata = {
                    'file_keys': keys,
                    'total_docs': len(keys),
                    'embedding_dim': embedding_dim,
                    'index_type': 'IndexFlatIP',
                    'created_at': datetime.now().isoformat(),
                    'intent': intent
                }
                
                if self.save_db and self.db_manager:
                    index_bytes = pickle.dumps(faiss.serialize_index(index))
                    index_name = f"faiss_{intent}"
                    
                    index_id = self.db_manager.save_faiss_index(
                        sub_id=sub_id,
                        index_name=index_name,
                        index_data=index_bytes,
                        metadata=metadata
                    )
                    
                    if index_id:
                        print(f"   ✅ {intent.upper()}: DB에 저장 (ID: {index_id})")
                else:
                    index_path = self.embedding_dir / f"faiss_{intent}.index"
                    faiss.write_index(index, str(index_path))
                    
                    metadata_path = self.embedding_dir / f"faiss_{intent}_metadata.json"
                    with open(metadata_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, ensure_ascii=False, indent=2)
                    
                    print(f"   ✅ {intent.upper()}: 파일로 저장")
            
            print("✅ 모든 의도별 FAISS 인덱스 구축 완료!")
            
        except Exception as e:
            print(f"❌ FAISS 인덱스 구축 실패: {e}")
            raise
        
    def generate_embeddings_ultra_fast(self, sub_id, user_prompt = None) -> None:
        print("\n" + "="*60)
        print("🚀 극한 최적화 임베딩 생성 시작")
        print("="*60)
        
        start_total_time = time.time()
        
        print("\n🔍 파일 변경사항 분석 중...")
        model = 'general' if self.use_local_model else 'gpt'
        new_files, deleted_files, modified_files, current_metadata = self.analyze_file_changes(sub_id, model)
        
        if not new_files and not deleted_files and not modified_files and not user_prompt:
            print("\n✅ 임베딩 변경사항이 없습니다.")
            
            if self.build_faiss:
                existing_embeddings, prompts = self.load_embeddings_from_file(sub_id, model)
                if self.file_extension == 'yaml':
                    current_files = []
                    for pattern in ['*.yaml', '*.yml']:
                        current_files.extend(list(self.docs_dir.rglob(pattern)))
                else:
                    current_files = list(self.docs_dir.rglob(f"*.{self.file_extension}"))
                
                self.rebuild_faiss_if_needed(sub_id, existing_embeddings, current_files, False, False)
            return
        
        embeddings, prompts = self.load_embeddings_from_file(sub_id, model)
        
        if deleted_files:
            embeddings = self.remove_deleted_embeddings(sub_id, deleted_files, embeddings)
        
        to_process_files = new_files + modified_files
        if to_process_files or user_prompt:
            print(f"\n🚀 {len(to_process_files)}개 파일 극한 최적화 처리 중...")
            
            if (self.use_local_model and 
                self.workers == 1 and 
                len(to_process_files) >= 10 and
                self.preload_all_texts
                and user_prompt):
                
                print("🚀🚀🚀 메가 배치 극한 최적화 모드 활성화!")
                
                file_data = self.preload_all_file_contents(to_process_files)
                
                if user_prompt:
                    print(f"\n🚀 사용자 추가 프롬프트 발견 {len(user_prompt)}개 파일 극한 최적화 처리 중...")
                    for k, v in user_prompt.items():
                        file_data[k] = v
                
                print(f"추가 임베딩 갯수: {len(file_data)}")
                new_embeddings = self.process_mega_batch(file_data)
                
                embeddings.update(new_embeddings)
                
                file_contents = {}
                if self.save_db:
                    for key in new_embeddings.keys():
                        if key in file_data:
                            file_contents[key] = file_data[key][1]
                
                print("\n💾 임베딩 저장 중...")
                if user_prompt:
                    self.write_embeddings_to_file(sub_id, new_embeddings, file_contents)
                else:
                    self.write_embeddings_to_file(sub_id, embeddings, file_contents)
                
            else:
                print("📝 표준 최적화 모드")
                
                to_process = []
                for file_path in to_process_files:
                    relative_path = file_path.relative_to(self.docs_dir)
                    file_key = str(relative_path)
                    to_process.append((file_path, file_key))
                
                file_contents = {}
                
                if self.use_local_model and self.workers == 1 and len(to_process) >= 4:
                    print("🚀 GPU 배치 처리 모드")
                    
                    batch_size = min(self.batch_size // 4, len(to_process))
                    success_count = 0
                    error_count = 0
                    
                    start_time = time.time()
                    
                    with tqdm(total=len(to_process), desc="GPU 배치 임베딩", ncols=100) as pbar:
                        for i in range(0, len(to_process), batch_size):
                            batch = to_process[i:i + batch_size]
                            batch_results = self.process_files_batch(batch)
                            
                            for file_key, embedding, error, content in batch_results:
                                if error:
                                    tqdm.write(f"   ❌ 오류 ({file_key}): {str(error)}")
                                    error_count += 1
                                else:
                                    embeddings[file_key] = embedding
                                    if content:
                                        file_contents[file_key] = content
                                    success_count += 1
                                
                                pbar.update(1)
                    
                    elapsed_time = time.time() - start_time
                    print(f"\n📊 GPU 배치 처리 결과:")
                    print(f"   - 성공: {success_count}개")
                    print(f"   - 실패: {error_count}개")
                    print(f"   - 처리 시간: {elapsed_time:.2f}초")
                    if elapsed_time > 0:
                        print(f"   - 평균 속도: {success_count/elapsed_time:.1f} 파일/초")
                    
                else:
                    workers = min(8, multiprocessing.cpu_count()) if self.use_local_model else min(50, self.api_workers)
                    delay = 0 if self.use_local_model else self.api_delay
                    
                    print(f"   - 워커 수: {workers}")
                    print(f"   - 처리 모드: {'로컬 모델' if self.use_local_model else 'GPT API'}")
                    
                    if user_prompt:
                        print(f"\n🚀 사용자 추가 프롬프트 발견 {len(user_prompt)}개")
                        for k, v in user_prompt.items():
                            to_process.append((v[1], k))
                            
                    process_func = partial(self.process_file, api_key=self.api_key, delay=delay)
                    
                    start_time = time.time()
                    success_count = 0
                    error_count = 0
                    new_embeddings = {}
                    
                    with tqdm(total=len(to_process), desc="임베딩 생성", ncols=100) as pbar:
                        with ProcessPoolExecutor(max_workers=workers) as executor:
                            futures = [executor.submit(process_func, file_info) for file_info in to_process]
                            
                            for future in as_completed(futures):
                                file_key, embedding, error, content = future.result()
                                
                                if error:
                                    tqdm.write(f"   ❌ 오류 ({file_key}): {str(error)}")
                                    error_count += 1
                                else:
                                    embeddings[file_key] = embedding
                                    new_embeddings[file_key] = embedding
                                    if content:
                                        file_contents[file_key] = content
                                    success_count += 1
                                
                                pbar.update(1)
                    
                    elapsed_time = time.time() - start_time
                    print(f"\n📊 처리 결과:")
                    print(f"   - 성공: {success_count}개")
                    print(f"   - 실패: {error_count}개")
                    print(f"   - 처리 시간: {elapsed_time:.2f}초")
                    
                    if elapsed_time > 0:
                        print(f"   - 평균 속도: {success_count/elapsed_time:.1f} 파일/초")
                
                print("\n💾 임베딩 저장 중...")
                if user_prompt:
                    self.write_embeddings_to_file(sub_id, new_embeddings, file_contents)
                else:
                    self.write_embeddings_to_file(sub_id, embeddings, file_contents)
        
        self.save_metadata(current_metadata)
        
        # BGE 모드 체크
        from config_utils import ConfigManager
        config = ConfigManager()
        use_bge = config.config.get('embedding', 'use_bge_mode', fallback='False')
        if use_bge.lower() == 'true':
            print("\n🧠 BGE M3 임베딩 모드: FAISS 인덱스 구축 건너뜀")
        else:
            if self.build_faiss and embeddings:
                print("\n🧠 FAISS 인덱스 구축 중...")
                if self.file_extension == 'yaml':
                    all_files = []
                    for pattern in ['*.yaml', '*.yml']:
                        all_files.extend(list(self.docs_dir.rglob(pattern)))
                else:
                    all_files = list(self.docs_dir.rglob(f"*.{self.file_extension}"))
                
                has_deletions = len(deleted_files) > 0
                has_modifications = len(modified_files) > 0 or len(new_files) > 0 or user_prompt
                self.rebuild_faiss_if_needed(sub_id, embeddings, all_files, has_deletions, has_modifications)
            
            if self.create_graph and (new_files or modified_files or deleted_files or user_prompt):
                print("\n📊 지식 그래프 업데이트 중...")
                self._update_knowledge_graph(sub_id, new_files + modified_files, deleted_files, user_prompt)
            
        total_time = time.time() - start_total_time
        print("\n" + "="*60)
        print("✅ 극한 최적화 임베딩 처리 완료!")
        print(f"📊 최종 통계:")
        print(f"   - 새로운 파일: {len(new_files)}개")
        print(f"   - 수정된 파일: {len(modified_files)}개")
        print(f"   - 삭제된 파일: {len(deleted_files)}개")
        print(f"   - 전체 임베딩: {len(embeddings)}개")
        print(f"   - 총 처리 시간: {total_time:.2f}초")
        if len(new_files + modified_files) > 0:
            speed = len(new_files + modified_files) / total_time
            print(f"   - 🚀 처리 속도: {speed:.1f} 파일/초")
        print("="*60)
    
    def _update_knowledge_graph(self, sub_id, updated_files: List[Path], deleted_files: List[str], user_prompt=None) -> None:
        # BGE 모드에서는 지식그래프 업데이트 건너뜀
        if not self.create_graph:
            print("   📋 BGE 모드: 지식그래프 업데이트 건너뜀")
            return

        try:
            if not hasattr(self, 'graph_generator') or self.graph_generator is None:
                if self.graph_dir is None:
                    self.graph_dir = self.embedding_dir.parent / "knowledge_graph"
                self.graph_dir.mkdir(parents=True, exist_ok=True)
                self.graph_generator = KnowledgeGraphGenerator(self.docs_dir, self.graph_dir)
            
            if self.file_extension == 'yaml':
                existing_files = []
                for pattern in ['*.yaml', '*.yml']:
                    existing_files.extend(list(self.docs_dir.rglob(pattern)))
            else:
                existing_files = list(self.docs_dir.rglob(f"*.{self.file_extension}"))
            
            self.graph_generator.process_incremental_update(existing_files, sub_id, self.save_db, user_prompt)
            print("   ✅ 지식 그래프 업데이트 완료")
            
        except Exception as e:
            print(f"   ❌ 지식 그래프 업데이트 실패: {e}")

    def update_single_file_in_faiss_all_subs(self, file_path: str, new_embedding: Embedding) -> Dict[str, bool]:
        """
        모든 구독키의 FAISS 인덱스에서 단일 파일의 임베딩 벡터 업데이트
        구독키 1번: 메타데이터 파일 기반
        구독키 2번+: DB 직접 조회 기반
        """
        print(f"   🔍 모든 구독키의 FAISS 인덱스 업데이트 중...")

        # 모든 구독키 찾기
        all_sub_ids = self.get_all_subscription_ids()
        results = {}

        for sub_id in all_sub_ids:
            print(f"   📌 구독키 {sub_id} FAISS 업데이트...")
            if sub_id == '1':
                # 구독키 1번: 기존 메타데이터 방식
                results[sub_id] = self.update_single_file_in_faiss(sub_id, file_path, new_embedding)
            else:
                # 구독키 2번+: DB 기반 방식
                results[sub_id] = self.update_single_file_in_faiss_from_db(sub_id, file_path, new_embedding)

        success_count = sum(1 for success in results.values() if success)
        print(f"   📊 FAISS 업데이트 결과: {success_count}/{len(all_sub_ids)} 성공")

        return results

    def update_single_file_in_faiss(self, sub_id: str, file_path: str, new_embedding: Embedding) -> bool:
        """
        특정 구독키의 FAISS 인덱스에서 단일 파일의 임베딩 벡터만 업데이트
        """
        try:
            model = 'general' if self.use_local_model else 'gpt'

            # 메타데이터 파일에서 파일 위치 찾기
            metadata_file = self.embedding_dir / f"metadata_{model}_{sub_id}.json"
            if not metadata_file.exists():
                print(f"     ❌ 메타데이터 파일이 없음: {metadata_file}")
                return False

            with open(metadata_file, 'r') as f:
                import json
                metadata = json.load(f)

            # 파일의 인덱스 위치 찾기 (두 가지 메타데이터 형식 지원)
            file_index = None
            file_keys = list(metadata.keys())

            for idx, key in enumerate(file_keys):
                if key == file_path:
                    file_index = idx
                    break

            if file_index is None:
                print(f"     ❌ FAISS 인덱스에서 파일을 찾을 수 없음: {file_path}")
                print(f"     📋 구독키 {sub_id} 메타데이터에 있는 파일들 (처음 5개):")
                for i, key in enumerate(file_keys[:5]):
                    print(f"       {i}: {key}")
                if len(file_keys) > 5:
                    print(f"       ... (총 {len(file_keys)}개 파일)")
                return False

            # 의도별 FAISS 인덱스 파일들 업데이트
            intent_types = ['general', 'qna', 'action', 'plan']
            file_intent = self._determine_file_intent(file_path)

            for intent in intent_types:
                faiss_file = self.embedding_dir / f"faiss_{intent}_{model}_{sub_id}.index"
                if faiss_file.exists():
                    # FAISS 인덱스 로드
                    import faiss
                    index = faiss.read_index(str(faiss_file))

                    if intent == file_intent:
                        # 해당 의도의 인덱스에서 벡터 업데이트
                        new_vector = self.to_vector_array([new_embedding.vector])
                        faiss.normalize_L2(new_vector)

                        # 기존 벡터를 새 벡터로 교체
                        index.remove_ids(np.array([file_index], dtype=np.int64))
                        index.add_with_ids(new_vector, np.array([file_index], dtype=np.int64))

                        # 인덱스 저장
                        faiss.write_index(index, str(faiss_file))
                        print(f"     ✅ 구독키 {sub_id} FAISS 인덱스 업데이트 완료: {intent}")

            # 메타데이터 업데이트 (두 가지 형식 지원)
            import hashlib
            file_path_obj = self.docs_dir / file_path
            if file_path_obj.exists():
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    content = f.read()
                new_checksum = hashlib.md5(content.encode()).hexdigest()

                # 기존 메타데이터 형식 확인
                if file_path in metadata:
                    existing_value = metadata[file_path]
                    if isinstance(existing_value, dict):
                        # 새로운 객체 형식 (구독키 2번 스타일)
                        import time
                        metadata[file_path].update({
                            'hash': new_checksum,
                            'size': len(content.encode('utf-8')),
                            'modified_time': int(time.time())
                        })
                    else:
                        # 기존 문자열 형식 (구독키 1번 스타일)
                        metadata[file_path] = new_checksum
                else:
                    # 파일이 없으면 기존 형식을 따라감 (구독키별 일관성 유지)
                    # 메타데이터의 첫 번째 항목 형식을 확인
                    if metadata and isinstance(next(iter(metadata.values())), dict):
                        # 객체 형식으로 추가
                        import time
                        metadata[file_path] = {
                            'path': file_path,
                            'hash': new_checksum,
                            'size': len(content.encode('utf-8')),
                            'modified_time': int(time.time()),
                            'created_time': int(time.time())
                        }
                    else:
                        # 문자열 형식으로 추가
                        metadata[file_path] = new_checksum

                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"     ✅ 구독키 {sub_id} 메타데이터 파일 업데이트 완료")

            return True

        except Exception as e:
            print(f"     ❌ 구독키 {sub_id} FAISS 업데이트 실패: {e}")
            return False

    def update_single_file_in_faiss_from_db(self, sub_id: str, file_path: str, new_embedding: Embedding) -> bool:
        """
        DB ai_faiss_indices와 ai_faiss_index_parts 테이블에서 파티션 데이터를 조회하여 FAISS 인덱스 업데이트 (구독키 2번+ 용)
        """
        try:
            if not hasattr(self, 'db_manager') or not self.db_manager:
                print(f"     ❌ DB Manager가 없어서 DB 기반 업데이트 불가")
                return False

            # ai_faiss_indices 테이블에서 해당 구독키의 FAISS 인덱스 조회
            model = 'general' if self.use_local_model else 'gpt'
            file_intent = self._determine_file_intent(file_path)
            index_name = f"faiss_{file_intent}"

            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 해당 구독키와 의도에 맞는 FAISS 인덱스 조회
                    query = """
                        SELECT index_data, metadata, artifact_metadata
                        FROM ai_faiss_indices
                        WHERE subscription_id = %s AND index_name = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                    """
                    cursor.execute(query, (sub_id, index_name))
                    faiss_row = cursor.fetchone()

            if not faiss_row:
                print(f"     📋 구독키 {sub_id}에 {index_name} FAISS 인덱스가 없어 생략: {file_path}")
                return True  # 인덱스가 없는 것은 오류가 아님

            # 메타데이터에서 파일 키 목록 파싱
            import json

            # artifact_metadata가 있으면 사용, 없으면 metadata 사용
            if faiss_row['artifact_metadata']:
                file_keys_data = json.loads(faiss_row['artifact_metadata'])
            else:
                file_keys_data = json.loads(faiss_row['metadata'])

            file_keys = file_keys_data.get('file_keys', [])

            # 파일의 인덱스 위치 찾기
            file_index = None
            for idx, key in enumerate(file_keys):
                if key == file_path:
                    file_index = idx
                    break

            if file_index is None:
                print(f"     📋 구독키 {sub_id} FAISS 인덱스에 파일 없어 생략: {file_path}")
                print(f"     📋 FAISS에 있는 파일들 (처음 5개):")
                for i, key in enumerate(file_keys[:5]):
                    print(f"       {i}: {key}")
                if len(file_keys) > 5:
                    print(f"       ... (총 {len(file_keys)}개 파일)")
                return True  # 파일이 없는 것은 오류가 아님

            # 파티션 데이터에서 FAISS 인덱스 로드
            faiss_index_data = self._load_partitioned_faiss_index(sub_id, index_name)

            if not faiss_index_data:
                print(f"     📋 구독키 {sub_id} FAISS 파티션 데이터가 비어있음 (파일은 메타데이터에 존재: 인덱스 {file_index})")
                return True  # 데이터가 없어도 오류는 아님

            # FAISS 인덱스 로드 및 업데이트
            import faiss
            import io

            # FAISS 인덱스 직접 로드 (bytes 데이터)
            if isinstance(faiss_index_data, bytes):
                index = faiss.deserialize_index(faiss_index_data)
            else:
                print(f"     ❌ 잘못된 데이터 타입: {type(faiss_index_data)}")
                return False

            # 새 벡터로 업데이트
            new_vector = self.to_vector_array([new_embedding.vector])
            faiss.normalize_L2(new_vector)

            # 기존 벡터를 새 벡터로 교체
            index.remove_ids(np.array([file_index], dtype=np.int64))
            index.add_with_ids(new_vector, np.array([file_index], dtype=np.int64))

            # 업데이트된 인덱스를 파티션으로 다시 저장
            updated_index_bytes = faiss.serialize_index(index)
            success = self._save_partitioned_faiss_index(sub_id, index_name, updated_index_bytes)

            if success:
                print(f"     ✅ 구독키 {sub_id} FAISS 인덱스 업데이트 완료: {index_name} (파티션 기반)")
            else:
                print(f"     ❌ 구독키 {sub_id} FAISS 파티션 저장 실패")

            return success

        except Exception as e:
            print(f"     ❌ 구독키 {sub_id} DB 기반 FAISS 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_partitioned_faiss_index(self, sub_id: str, index_name: str) -> bytes:
        """
        ai_faiss_index_parts에서 파티션 데이터를 로드하여 완전한 FAISS 인덱스 복원
        """
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 파티션 데이터 조회
                    query = """
                        SELECT part_no, chunk
                        FROM ai_faiss_index_parts
                        WHERE subscription_id = %s AND index_name = %s
                        ORDER BY part_no
                    """
                    cursor.execute(query, (sub_id, index_name))
                    parts = cursor.fetchall()

            if not parts:
                return b''

            # 파티션들을 순서대로 합치기
            combined_data = b''
            for part in parts:
                combined_data += part['chunk']

            return combined_data

        except Exception as e:
            print(f"     ❌ FAISS 파티션 로드 실패: {e}")
            return b''

    def _save_partitioned_faiss_index(self, sub_id: str, index_name: str, index_data: bytes) -> bool:
        """
        FAISS 인덱스 데이터를 파티션으로 분할하여 ai_faiss_index_parts에 저장
        """
        try:
            # 기존 파티션 삭제
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    delete_query = """
                        DELETE FROM ai_faiss_index_parts
                        WHERE subscription_id = %s AND index_name = %s
                    """
                    cursor.execute(delete_query, (sub_id, index_name))

                    # 새 파티션 저장 (1MB 청크로 분할)
                    chunk_size = 1024 * 1024  # 1MB
                    part_no = 0

                    for i in range(0, len(index_data), chunk_size):
                        chunk = index_data[i:i + chunk_size]

                        insert_query = """
                            INSERT INTO ai_faiss_index_parts (subscription_id, index_name, part_no, chunk)
                            VALUES (%s, %s, %s, %s)
                        """
                        cursor.execute(insert_query, (sub_id, index_name, part_no, chunk))
                        part_no += 1

                    conn.commit()
                    print(f"     💾 FAISS 인덱스 {part_no}개 파티션으로 저장 완료")

            return True

        except Exception as e:
            print(f"     ❌ FAISS 파티션 저장 실패: {e}")
            return False

    def _load_partitioned_knowledge_graph(self, sub_id: str, data_name: str) -> Optional[str]:
        """파티션으로 저장된 지식그래프 데이터 로드"""
        try:
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # kg_id 먼저 조회
                    kg_query = """
                        SELECT id FROM ai_knowledge_graph
                        WHERE subscription_id = %s AND data_name = %s
                    """
                    cursor.execute(kg_query, (sub_id, data_name))
                    kg_result = cursor.fetchone()

                    if not kg_result:
                        return None

                    kg_id = kg_result['id']

                    # 파티션 데이터 조회 (part_no 순으로)
                    query = """
                        SELECT chunk, part_no
                        FROM ai_knowledge_graph_parts
                        WHERE kg_id = %s
                        ORDER BY part_no
                    """
                    cursor.execute(query, (kg_id,))
                    parts = cursor.fetchall()

                    if not parts:
                        return None

                    # 파티션 데이터 결합 (바이너리 데이터를 텍스트로 변환)
                    combined_data = b""
                    for part in parts:
                        combined_data += part['chunk']

                    return combined_data.decode('utf-8')

        except Exception as e:
            print(f"❌ 지식그래프 파티션 로드 실패 ({sub_id}, {data_name}): {e}")
            return None

    def _save_partitioned_knowledge_graph(self, sub_id: str, data_name: str, data_content: str) -> bool:
        """지식그래프 데이터를 파티션으로 분할해서 저장"""
        try:
            CHUNK_SIZE = 1024 * 1024  # 1MB 청크

            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # kg_id 먼저 조회
                    kg_query = """
                        SELECT id FROM ai_knowledge_graph
                        WHERE subscription_id = %s AND data_name = %s
                    """
                    cursor.execute(kg_query, (sub_id, data_name))
                    kg_result = cursor.fetchone()

                    if not kg_result:
                        print(f"  ❌ 지식그래프 메인 레코드가 없음: {sub_id}, {data_name}")
                        return False

                    kg_id = kg_result['id']

                    # 기존 파티션 삭제
                    delete_query = """
                        DELETE FROM ai_knowledge_graph_parts
                        WHERE kg_id = %s
                    """
                    cursor.execute(delete_query, (kg_id,))

                    # 새 파티션으로 분할 저장
                    data_bytes = data_content.encode('utf-8')
                    total_size = len(data_bytes)
                    part_no = 0

                    for i in range(0, total_size, CHUNK_SIZE):
                        chunk = data_bytes[i:i + CHUNK_SIZE]

                        insert_query = """
                            INSERT INTO ai_knowledge_graph_parts
                            (kg_id, part_no, chunk)
                            VALUES (%s, %s, %s)
                        """
                        cursor.execute(insert_query, (kg_id, part_no, chunk))
                        part_no += 1

                    print(f"  📦 지식그래프 {data_name} 파티션 저장 완료: {part_no}개 파트")
                    return True

        except Exception as e:
            print(f"❌ 지식그래프 파티션 저장 실패 ({sub_id}, {data_name}): {e}")
            return False

    def update_single_file_in_knowledge_graph_all_subs(self, file_path: str) -> Dict[str, bool]:
        """
        모든 구독키의 지식 그래프에서 단일 파일 정보 업데이트
        구독키 1번: 기존 방식
        구독키 2번+: DB 기반 방식 (하지만 지식그래프는 공통이므로 한 번만 처리)
        """
        print(f"   🕸️  모든 구독키의 지식 그래프 업데이트 중...")

        # 모든 구독키 찾기
        all_sub_ids = self.get_all_subscription_ids()
        results = {}

        for sub_id in all_sub_ids:
            print(f"   📌 구독키 {sub_id} 지식그래프 업데이트...")
            if sub_id == '1':
                # 구독키 1번: 기존 방식
                results[sub_id] = self.update_single_file_in_knowledge_graph(sub_id, file_path)
            else:
                # 구독키 2번+: DB 기반 확인 후 업데이트
                results[sub_id] = self.update_single_file_in_knowledge_graph_from_db(sub_id, file_path)

        success_count = sum(1 for success in results.values() if success)
        print(f"   📊 지식그래프 업데이트 결과: {success_count}/{len(all_sub_ids)} 성공")

        return results

    def update_single_file_in_knowledge_graph(self, sub_id: str, file_path: str) -> bool:
        """
        특정 구독키의 지식 그래프에서 단일 파일의 정보만 업데이트
        """
        # BGE 모드에서는 지식그래프 업데이트 건너뜀
        if not self.create_graph:
            print(f"     📋 BGE 모드: 구독키 {sub_id} 지식그래프 업데이트 건너뜀")
            return True  # BGE 모드에서는 성공으로 처리

        try:
            if not hasattr(self, 'graph_generator') or self.graph_generator is None:
                if self.graph_dir is None:
                    self.graph_dir = self.embedding_dir.parent / "knowledge_graph"
                self.graph_dir.mkdir(parents=True, exist_ok=True)
                from aibot_knowledge_graph import KnowledgeGraphGenerator
                self.graph_generator = KnowledgeGraphGenerator(self.docs_dir, self.graph_dir)

            # 단일 파일 경로를 Path 객체로 변환
            file_path_obj = self.docs_dir / file_path
            if not file_path_obj.exists():
                print(f"     ❌ 파일이 존재하지 않음: {file_path_obj}")
                return False

            # 단일 파일에 대해 지식 그래프 업데이트
            updated_files = [file_path_obj]
            self.graph_generator.process_incremental_update(updated_files, sub_id, self.save_db)
            print(f"     ✅ 구독키 {sub_id} 지식 그래프 단일 파일 업데이트 완료: {file_path}")

            return True

        except Exception as e:
            print(f"     ❌ 구독키 {sub_id} 지식 그래프 업데이트 실패: {e}")
            return False

    def update_single_file_in_knowledge_graph_from_db(self, sub_id: str, file_path: str) -> bool:
        """
        DB ai_knowledge_graph 테이블 기반으로 지식 그래프 업데이트 (구독키 2번+ 용)
        분할 저장된 지식그래프 데이터를 로드하여 업데이트 수행
        """
        # BGE 모드에서는 지식그래프 업데이트 건너뜀
        if not self.create_graph:
            print(f"     📋 BGE 모드: 구독키 {sub_id} DB 지식그래프 업데이트 건너뜀")
            return True  # BGE 모드에서는 성공으로 처리

        try:
            if not hasattr(self, 'db_manager') or not self.db_manager:
                print(f"     ❌ DB Manager가 없어서 DB 기반 업데이트 불가")
                return False

            # ai_knowledge_graph 테이블에서 해당 구독키의 지식그래프 데이터 존재 확인
            with self.db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 해당 구독키의 지식그래프 데이터가 있는지 확인
                    query = """
                        SELECT COUNT(*) as count
                        FROM ai_knowledge_graph
                        WHERE subscription_id = %s
                    """
                    cursor.execute(query, (sub_id,))
                    result = cursor.fetchone()

            if result['count'] == 0:
                print(f"     📋 구독키 {sub_id}에 지식그래프 데이터가 없어 업데이트 생략: {file_path}")
                return True  # 데이터가 없는 것은 오류가 아님

            # 파일이 물리적으로 존재하는지 확인
            file_path_obj = self.docs_dir / file_path
            if not file_path_obj.exists():
                print(f"     ❌ 파일이 존재하지 않음: {file_path_obj}")
                return False

            # 지식 그래프 생성기 초기화
            if not hasattr(self, 'graph_generator'):
                self.graph_dir.mkdir(parents=True, exist_ok=True)
                from aibot_knowledge_graph import KnowledgeGraphGenerator
                self.graph_generator = KnowledgeGraphGenerator(self.docs_dir, self.graph_dir)

            # DB에서 지식그래프 데이터 로드 (기존 파티션 처리 방식 사용)
            try:
                nodes_data = self.db_manager.load_knowledge_graph(subscription_id=sub_id, data_name="nodes_data")
                relations_data = self.db_manager.load_knowledge_graph(subscription_id=sub_id, data_name="relations_data")
                records_data = self.db_manager.load_knowledge_graph(subscription_id=sub_id, data_name="records_data")
                term_mapping = self.db_manager.load_knowledge_graph(subscription_id=sub_id, data_name="term_mapping")

                print(f"     📦 지식그래프 데이터 로드 완료 (DB 기반)")

                # JSON 파싱
                import json
                knowledge_data = {
                    'nodes': json.loads(nodes_data),
                    'relations': json.loads(relations_data),
                    'records': json.loads(records_data),
                    'terms': json.loads(term_mapping)
                }

                # 지식그래프에 기존 데이터 로드
                self.graph_generator.knowledge_graph.load_from_db(knowledge_data)

                # 파일에서 기존 데이터 제거 후 새 데이터로 업데이트
                self.graph_generator.knowledge_graph.remove_db_data(file_path)

                # 단일 파일 처리를 위한 프로세스 실행
                with open(file_path_obj, 'r', encoding='utf-8') as f:
                    file_content = f.read()

                process_files = {file_path: file_content}
                self.graph_generator.process_update(process_files, sub_id, self.save_db)

                print(f"     ✅ 지식그래프 업데이트 완료")

            except Exception as kg_error:
                print(f"     ⚠️  지식그래프 기반 업데이트 실패, 증분 업데이트 사용: {kg_error}")
                # 증분 업데이트로 fallback
                updated_files = [file_path_obj]
                self.graph_generator.process_incremental_update(updated_files, sub_id, self.save_db)

            print(f"     ✅ 구독키 {sub_id} 지식 그래프 단일 파일 업데이트 완료: {file_path} (DB 기반)")
            return True

        except Exception as e:
            print(f"     ❌ 구독키 {sub_id} DB 기반 지식 그래프 업데이트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _determine_file_intent(self, file_path: str) -> str:
        """파일 경로를 기반으로 의도 타입 결정"""
        if 'action/' in file_path:
            return 'action'
        elif 'plan/' in file_path:
            return 'plan'
        elif 'qna/' in file_path:
            return 'qna'
        else:
            return 'general'

    def get_all_subscription_ids(self) -> List[str]:
        """
        시스템에서 사용 중인 모든 구독키 찾기
        임베딩 디렉토리의 메타데이터 파일들을 확인하여 구독키 추출
        """
        subscription_ids = set()

        try:
            # 현재 모델 타입 확인
            model = 'general' if self.use_local_model else 'gpt'

            # 임베딩 디렉토리에서 메타데이터 파일들 찾기
            # 패턴: metadata_{model}_{sub_id}.json
            import glob
            pattern = str(self.embedding_dir / f"metadata_{model}_*.json")
            metadata_files = glob.glob(pattern)

            for file_path in metadata_files:
                # 파일명에서 구독키 추출
                filename = Path(file_path).stem  # metadata_general_1
                parts = filename.split('_')
                if len(parts) >= 3:
                    sub_id = parts[-1]  # 마지막 부분이 구독키
                    subscription_ids.add(sub_id)

            # DB에서도 구독키 확인 (추가 검증)
            if hasattr(self, 'db_manager') and self.db_manager:
                try:
                    with self.db_manager.get_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT DISTINCT subscription_id FROM openai_prompts")
                            db_sub_ids = [str(row['subscription_id']) for row in cursor.fetchall()]  # 문자열로 변환
                            subscription_ids.update(db_sub_ids)
                except Exception as e:
                    print(f"   ⚠️ DB에서 구독키 조회 중 오류: {e}")

            result = sorted(list(subscription_ids), key=lambda x: int(x) if x.isdigit() else float('inf'))
            print(f"   📋 감지된 구독키: {result}")
            return result

        except Exception as e:
            print(f"   ❌ 구독키 탐지 실패: {e}")
            return ['1']  # 기본값으로 구독키 1번만 반환

    def set_ultra_performance_profile(self):
        if self.use_local_model and torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory >= 24:
                self.workers = 1
                self.batch_size = 128
                self.mega_batch_size = 256
                self.preload_all_texts = True
                print("🚀 RTX 5090/4090 극한 최적화 프로파일")
            elif gpu_memory >= 12:
                self.workers = 1
                self.batch_size = 64
                self.mega_batch_size = 128
                self.preload_all_texts = True
                print("🚀 RTX 4070Ti/3080 고속 프로파일")
            elif gpu_memory >= 8:
                self.workers = 1
                self.batch_size = 32
                self.mega_batch_size = 64
                self.preload_all_texts = False
                print("🚀 RTX 4060Ti/3070 최적화 프로파일")
            
            print(f"   - 워커: {self.workers}개")
            print(f"   - 배치: {self.batch_size}개")  
            print(f"   - 메가배치: {self.mega_batch_size}개")
            print(f"   - 전체 로딩: {'활성화' if self.preload_all_texts else '비활성화'}")

    def set_performance_profile(self, profile: str = "ultra"):
        profiles = {
                "conservative": (4, 10, 0.1, 20),
                "balanced": (8, 20, 0.05, 50),
                "aggressive": (12, 50, 0.01, 100),
                "ultra": (16, 100, 0.001, 200),
                "rtx5090": (12, 16, 0, 256)
        }
        
        if profile in profiles:
            self.workers, self.api_workers, self.api_delay, self.batch_size = profiles[profile]
            print(f"🎯 성능 프로파일 설정: {profile}")
            print(f"   - 로컬 워커: {self.workers}개")
            print(f"   - API 워커: {self.api_workers}개")
            print(f"   - API 딜레이: {self.api_delay}초")
            print(f"   - 배치 크기: {self.batch_size}개")


    def embed_changes_only(self, sub_id, user_prompt=None, mode=None, tags=None):
        print("\n" + "="*60)
        print("🚀 [STEP 1/4] 임베딩 단계 시작")
        print("="*60)

        start_total_time = time.time()
        model = 'general' if self.use_local_model else 'gpt'

        print("\n🔍 파일 변경사항 분석 중...")
        new_files, deleted_files, modified_files, current_metadata = self.analyze_db_changes(sub_id, model)

        to_process_files = new_files + modified_files

        if not to_process_files and not user_prompt and not deleted_files:
            print("\n✅ 임베딩 변경사항이 없습니다.")

            self.save_db_metadata(sub_id, model)

            return {
                "new_embedding": {},
                "all_files": [],
                "has_deletions": False,
                "has_modifications": False,
                "new_files": new_files,
                "modified_files": modified_files,
                "deleted_files": deleted_files,
                "user_prompt": user_prompt,
                "tags": tags,
            }

        print(f"\n🚀 {len(to_process_files)}개 파일 임베딩 처리 준비...")

        if deleted_files:
            print(f"\n🗑️ 삭제된 파일들 DB에서 제거 중... ({len(deleted_files)}개)")
            temp_embeddings = {}
            self.remove_deleted_embeddings(sub_id, deleted_files, temp_embeddings)

        def _normalize_user_prompts(up):
            if not up:
                return {}
            norm = {}
            for k, v in up.items():
                if isinstance(v, tuple):
                    parts = [p for p in v if isinstance(p, str)]
                    chosen = None
                    for p in parts:
                        if p.lstrip().startswith(("type:", "---")):
                            chosen = p; break
                    if not chosen:
                        chosen = "\n\n".join(parts) if parts else str(v)
                    norm[k] = chosen
                elif isinstance(v, str):
                    norm[k] = v
                else:
                    norm[k] = str(v)
            return norm

        user_prompt_norm = _normalize_user_prompts(user_prompt)

        file_contents = {}
        new_embeddings = {}
        
        if (self.use_local_model and self.workers == 1 and len(to_process_files) >= 10
            and self.preload_all_texts and (to_process_files or user_prompt_norm)):
            print("🚀🚀🚀 메가 배치 극한 최적화 모드 활성화!")

            file_data = self.preload_all_file_contents(to_process_files)
            
            for k, v in user_prompt_norm.items():
                file_data[k] = (None, v)

            print(f"추가 임베딩 갯수: {len(file_data)}")

            new_embeddings = self.process_mega_batch(file_data)

            if self.save_db:
                for key, pair in file_data.items():
                    if isinstance(pair, (tuple, list)) and len(pair) >= 2:
                        file_contents[key] = pair[1]

        else:
            print("📝 표준 최적화 모드")
            to_process = []

            for relative_path in to_process_files:
                file_path = f"{self.docs_dir}/{relative_path}"
                file_key = str(relative_path)
                to_process.append((file_path, file_key))

            if user_prompt_norm:
                print(f"\n🚀 사용자 추가 프롬프트 {len(user_prompt_norm)}개 처리")
                for k, v in user_prompt_norm.items():
                    to_process.append((v, k))

            if self.use_local_model and self.workers == 1 and len(to_process) >= 4:
                print("🚀 GPU 배치 처리 모드")
                batch_size = min(self.batch_size // 4, len(to_process))
                success_count = 0; error_count = 0
                start_time = time.time()

                from tqdm import tqdm
                with tqdm(total=len(to_process), desc="GPU 배치 임베딩", ncols=100) as pbar:
                    for i in range(0, len(to_process), batch_size):
                        batch = to_process[i:i + batch_size]
                        batch_results = self.process_files_batch(batch)
                        for file_key, embedding, error, content in batch_results:
                            if error:
                                tqdm.write(f"   ❌ 오류 ({file_key}): {str(error)}")
                                error_count += 1
                            else:
                                new_embeddings[file_key] = embedding
                                if content: file_contents[file_key] = content
                                success_count += 1
                            pbar.update(1)

                elapsed_time = time.time() - start_time
                print(f"\n📊 GPU 배치 처리 결과: 성공 {success_count}, 실패 {error_count}, {elapsed_time:.2f}초")

            else:
                workers = min(8, multiprocessing.cpu_count()) if self.use_local_model else min(50, self.api_workers)
                delay = 0 if self.use_local_model else self.api_delay
                print(f"   - 워커 수: {workers}")
                print(f"   - 처리 모드: {'로컬 모델' if self.use_local_model else 'GPT API'}")

                from functools import partial
                from concurrent.futures import ProcessPoolExecutor, as_completed
                from tqdm import tqdm

                process_func = partial(self.process_file, api_key=self.api_key, delay=delay)
                start_time = time.time()
                success_count = 0; error_count = 0

                with tqdm(total=len(to_process), desc="임베딩 생성", ncols=100) as pbar:
                    with ProcessPoolExecutor(max_workers=workers) as executor:
                        futures = [executor.submit(process_func, fi) for fi in to_process]
                        for fut in as_completed(futures):
                            file_key, embedding, error, content = fut.result()
                            if error:
                                tqdm.write(f"   ❌ 오류 ({file_key}): {str(error)}")
                                error_count += 1
                            else:
                                new_embeddings[file_key] = embedding
                                if content: file_contents[file_key] = content
                                success_count += 1
                            pbar.update(1)

                elapsed_time = time.time() - start_time
                print(f"\n📊 처리 결과: 성공 {success_count}, 실패 {error_count}, {elapsed_time:.2f}초")

        print("\n💾 임베딩 저장 중...")
        if user_prompt_norm:
            self.write_embeddings_to_file(sub_id, new_embeddings, file_contents, tags)

        print("\n🔍 임베딩 후 파일 변경 사항 재분석 중...")
        new_files, deleted_files, modified_files, current_metadata = self.analyze_db_changes(sub_id, model)

        self.save_db_metadata(sub_id, model)

        if self.file_extension == 'yaml':
            all_files = []
            for pattern in ['*.yaml', '*.yml']:
                all_files.extend(list(self.docs_dir.rglob(pattern)))
        else:
            all_files = list(self.docs_dir.rglob(f"*.{self.file_extension}"))

        has_deletions = len(deleted_files) > 0
        has_modifications = (len(modified_files) > 0 or len(new_files) > 0 or bool(user_prompt_norm))

        total_time = time.time() - start_total_time
        print("\n" + "="*60)
        print("✅ [STEP 1/2] 임베딩 단계 완료")
        print(f"   - 전체 임베딩: {len(new_embeddings)}개")
        print(f"   - 처리 시간: {total_time:.2f}초")
        print("="*60)

        return {
            "new_embedding": new_embeddings,
            "all_files": all_files,
            "has_deletions": has_deletions,
            "has_modifications": has_modifications,
            "new_files": new_files,
            "modified_files": modified_files,
            "deleted_files": deleted_files,
            "user_prompt": user_prompt_norm,
            "tags": tags,
        }
    
    def postprocess_with_faiss_and_kg(self, sub_id, ctx: dict, initial_run = False):
        model = 'general' if self.use_local_model else 'gpt'
        embeddings, prompts = self.load_embeddings_from_file(sub_id, model)

        all_files = ctx["all_files"]
        has_deletions = ctx["has_deletions"]
        has_modifications = ctx["has_modifications"]
        new_files = ctx["new_files"]
        modified_files = ctx["modified_files"]
        deleted_files = ctx["deleted_files"]
        user_prompt = ctx["user_prompt"]

        print(f"new_files : [{new_files}]")
        print(f"modified_files : [{modified_files}]")
        print(f"deleted_files : [{deleted_files}]")

        if deleted_files and embeddings:
            print(f"\n🗑️ 삭제된 파일들의 임베딩 제거 중... ({len(deleted_files)}개)")
            embeddings = self.remove_deleted_embeddings(sub_id, deleted_files, embeddings)

        # BGE 모드 확인 (기존 클래스 변수 사용)
        from config_utils import ConfigManager
        config = ConfigManager()
        is_bge_mode = config.config.get('embedding', 'use_bge_mode', fallback='False')
        print(f"is_bge_mode: {is_bge_mode}")
        if is_bge_mode:
            print("🔍 BGE 모드 감지 - FAISS 인덱스 및 지식그래프 구축을 건너뜁니다")
            print("   BGE 모드에서는 자체 벡터 시스템을 사용합니다")

        if initial_run:
            if self.build_faiss and embeddings:
                print(f"\n🧠 [STEP 2/4] FAISS 인덱스 구축 중... ")
                self.build_faiss_indexes(sub_id, embeddings)
            elif is_bge_mode:
                print(f"\n⚠️ [STEP 2/4] BGE 모드로 인해 FAISS 인덱스 구축 생략")
        else:
            if self.build_faiss and embeddings:
                print(f"\n🧠 [STEP 2/4] FAISS 인덱스 구축 중... ")
                self.rebuild_faiss_if_needed(sub_id, embeddings, all_files, has_deletions, has_modifications)
            elif is_bge_mode:
                print(f"\n⚠️ [STEP 2/4] BGE 모드로 인해 FAISS 인덱스 재구축 생략")

        # BGE 모드일 때는 지식그래프도 건너뛰기 (self.create_graph가 이미 False로 설정됨)
        if is_bge_mode:
            print(f"\n⚠️ [STEP 3/4] BGE 모드로 인해 지식그래프 처리도 생략")
        elif self.create_graph and initial_run:
            print("\n📊 [STEP 3/4] 지식 그래프 생성 중...")
            self.graph_generator = KnowledgeGraphGenerator(self.docs_dir, self.graph_dir)
            self.graph_generator.process_update(prompts, sub_id, self.save_db)
        else:
            if self.create_graph and (new_files or modified_files or deleted_files):
                print("\n📊 [STEP 3/4] 지식 그래프 업데이트 중...")
                
                nodes_data = self.db_manager.load_knowledge_graph(subscription_id=sub_id, data_name="nodes_data")
                relations_data = self.db_manager.load_knowledge_graph(subscription_id=sub_id, data_name="relations_data")
                records_data = self.db_manager.load_knowledge_graph(subscription_id=sub_id, data_name="records_data")
                term_mapping = self.db_manager.load_knowledge_graph(subscription_id=sub_id, data_name="term_mapping")
                
                knowledge_data = {
                    'nodes': json.loads(nodes_data),
                    'relations': json.loads(relations_data),
                    'records': json.loads(records_data),
                    'terms': json.loads(term_mapping)
                }
                
                self.graph_generator = KnowledgeGraphGenerator(self.docs_dir, self.graph_dir)
                self.graph_generator.knowledge_graph.load_from_db(knowledge_data) 
            
                process_files = {}
                
                if len(new_files) > 0:
                    for k in new_files:
                        process_files.update({k: prompts[k]})
                    
                if len(modified_files) > 0 :
                    for k in modified_files:
                        self.graph_generator.knowledge_graph.remove_db_data(k)
                        process_files.update({k: prompts[k]})
                    
                if len(deleted_files) > 0 :
                    for k in deleted_files:
                        self.graph_generator.knowledge_graph.remove_db_data(k)
                
                self.graph_generator.process_update(process_files, sub_id, self.save_db)

        if is_bge_mode:
            print("\n✅ [STEP 4/4] BGE 모드로 인해 FAISS/지식그래프 처리 생략 완료")
        else:
            print("\n✅ [STEP 4/4] 사후처리 단계 완료")

def init(sub_id):
    save_db = True
    use_local_model = True
    local_model_path = "/data/models/bge-m3"
    batch_db_operations = True
    
    if use_local_model:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    generator = EmbeddingGenerator(
        save_db=save_db,
        use_local_model=use_local_model,
        local_model_path=local_model_path if use_local_model else None,
        batch_db_operations=batch_db_operations
    )
    
    generator.set_ultra_performance_profile()
    generator.generate_embeddings_ultra_fast(sub_id)
    
def main():
    save_db = True
    use_local_model = True
    local_model_path = "/data/models/bge-m3"
    batch_db_operations = True
    
    if use_local_model:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    generator = EmbeddingGenerator(
        save_db=save_db,
        use_local_model=use_local_model,
        local_model_path=local_model_path if use_local_model else None,
        batch_db_operations=batch_db_operations
    )
    
    generator.set_ultra_performance_profile()
    
    '''file_data = {
        "qna/query-order-command2.yaml": 
        ('질문: order command syntax and usage 답변: order command sorts the specific fields to be printed out in the specified order, and displays the remaining fields in lexicographical order.\n\nSyntax\n\n\norder FIELD, ...\n\n\nFIELD, ...\nNames of fields to be ordered in order, separated by a comma (,). The command sorts the fields that are not listed here in lexicographical order.\n\nUsage\n\n1. Define the field output order of the sys_cpu_logs table as *kernel*, *idle*, *user*, *_time*, *_table*, and *_id*.\n\n\ntable sys_cpu_logs | order kernel, idle, user, _time, _table, _id\n\n\n2. Define the field output order of the sys_cpu_logs table as *idle*, *kernel* and then the rest in lexicographical order\n\n\ntable sys_cpu_logs | order idle, kernel\n 관련키워드: 쿼리 order 명령어 query 오더 command 쿼리 오더 명령어 query order 명령어 query 오더 명령어 쿼리 order command query order command 쿼리 오더 command',
        'type: qna\ndescription: null\nname: query-order-command\nenabled: true\napp_code: null\nquestion: order command syntax and usage\nanswer: |\n  order command sorts the specific fields to be printed out in the specified order, and displays the remaining fields in lexicographical order.\n\n  Syntax\n\n  \n  order FIELD, ...\n  \n\n  FIELD, ...\n  Names of fields to be ordered in order, separated by a comma (,). The command sorts the fields that are not listed here in lexicographical order.\n\n  Usage\n\n  1. Define the field output order of the sys_cpu_logs table as *kernel*, *idle*, *user*, *_time*, *_table*, and *_id*.\n\n  \n  table sys_cpu_logs | order kernel, idle, user, _time, _table, _id\n  \n\n  2. Define the field output order of the sys_cpu_logs table as *idle*, *kernel* and then the rest in lexicographical order\n\n  \n  table sys_cpu_logs | order idle, kernel\n  \naliases:\n- 쿼리 order 명령어\n- query 오더 command\n- 쿼리 오더 명령어\n- query order 명령어\n- query 오더 명령어\n- 쿼리 order command\n- query order command\n- 쿼리 오더 command\n')
    }'''
    
    new_sub_id = 2
    ctx = generator.embed_changes_only(new_sub_id)
    generator.postprocess_with_faiss_and_kg(new_sub_id, ctx)

if __name__ == "__main__":
    main()