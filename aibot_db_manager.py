import pymysql
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid
import logging
import pickle
import base64
import faiss, io, zlib, hashlib, math
from aibot_db_command import TABLE_QUERIES
from contextlib import contextmanager
from config_utils import ConfigManager
from aibot_prompts_class import Record, LlmPrompt, OpenAiPrompt

config = ConfigManager()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AibotDBManager:
    def __init__(self, config: Dict[str, Any], query_properties: Dict[str, str]):
        self.config = config
        self.queries = query_properties
        self.TABLE_QUERIES = TABLE_QUERIES

        # Question 벡터를 위한 쿼리 추가
        if 'INSERT_PROMPT' not in self.queries:
            self.queries['INSERT_PROMPT'] = """
                INSERT INTO openai_prompts (
                    guid, subscription_id, name, description, type,
                    prompt, token_count, vector, question_vector,
                    source, tags, enabled, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, 1, NOW(), NOW()
                )
            """

        if 'UPDATE_PROMPT' not in self.queries:
            self.queries['UPDATE_PROMPT'] = """
                UPDATE openai_prompts SET
                    name = %s, description = %s, type = %s,
                    prompt = %s, token_count = %s, vector = %s,
                    question_vector = %s,
                    source = %s, tags = %s, updated_at = NOW()
                WHERE guid = %s
            """

        self.connection_pool = []
        self.CHUNK_SIZE = 4 * 1024 * 1024
        self.METADATA_COMPRESS_THRESHOLD = 64 * 1024
        
    @contextmanager
    def get_connection(self):
        connection = None
        try:
            db_config = self.config.get_db_config()

            connection = pymysql.connect(
                host=db_config.get('url', 'localhost'),
                port=db_config.get('port', 3306),
                user=db_config.get('user'),
                password=db_config.get('password'),
                database=db_config.get('database'),
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor,
                binary_prefix=True
            )
            yield connection
        except Exception as e:
            import traceback
            logger.error(f"DB 연결 오류: {e}")
            logger.error(f"스택 트레이스:\n{traceback.format_exc()}")
            raise
        finally:
            if connection:
                connection.close()
    
    def initialize_tables(self) -> bool:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    for table_name in self.TABLE_QUERIES.keys():
                        try:
                            check_query = self.queries.get('CHECK_TABLES')
                            if not check_query:
                                raise ValueError("CHECK_TABLES query is None")

                            cursor.execute(check_query, (table_name,))
                            exists = cursor.fetchone()
                            exists = exists[0] if not isinstance(exists, dict) else list(exists.values())[0]

                            if exists == 0:
                                cursor.execute(self.TABLE_QUERIES.get(table_name))

                        except Exception as e:
                            logger.error(f"테이블 확인/생성 실패 [{table_name}]: {e}")
                            raise

                    conn.commit()
                    return True

        except Exception as e:
            logger.error(f"테이블 초기화 실패: {e}")
            return False
    
    def get_all_datas(self, sub_id, model=None):
        embedding_data = self.get_prompts(s_id=sub_id, model=model)
        
        def decode_vector(v):
            if isinstance(v, str):
                try:
                    v = base64.b64decode(v)
                except Exception as e:
                    print("base64 decoding failed:", e)
                    return None

            try:
                return np.frombuffer(v, dtype=np.float32)
            except Exception as e:
                print(f"Error 발생한 벡터 : {v}")
                return None
        
        embedding = {}
        for data in embedding_data:
            source_file = data['source']

            decoded_vector = decode_vector(data['vector'])
            if decoded_vector is None:
                print(f"❌ 벡터 디코딩 실패한 파일: {source_file} (ID: {data.get('id', 'Unknown')})")
                continue

            # Question 벡터 디코딩 추가
            decoded_question_vector = None
            if data.get('question_vector'):
                decoded_question_vector = decode_vector(data['question_vector'])
                if decoded_question_vector is None:
                    print(f"⚠️ Question 벡터 디코딩 실패: {source_file}")

            embedding[source_file] = {
                "token_count": data['token_count'],
                "vector": decoded_vector,
                "question_vector": decoded_question_vector  # Question 벡터 추가
            }
        
        faiss_data = self.load_all_faiss_index(sub_id=sub_id)
        f_data = {}
        for data in faiss_data:
            intent = data.get('intent', '<unknown>')
            try:
                raw_index = data.get('index_data')
                if isinstance(raw_index, tuple):
                    raw_index = raw_index[0]  
                if not isinstance(raw_index, (bytes, bytearray)):
                    logger.error(f"[{intent}] index_data 타입 오류: {type(raw_index)}")
                    continue

                try:
                    serialized = pickle.loads(raw_index)
                    index = faiss.deserialize_index(serialized)
                except Exception as e:
                    logger.error(f"[{intent}] pickle.loads 실패: {e}")
                    continue

                try:
                    metadata = data.get('metadata', {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                except Exception as e:
                    logger.error(f"[{intent}] metadata 처리 실패: {e}")
                    metadata = {}

                f_data[intent] = {
                    'index': index,
                    'metadata': metadata
                }
            except Exception as e:
                logger.exception(f"[{intent}] f_data 구성 중 알 수 없는 오류 발생")
 
        nodes_data = self.load_knowledge_graph(subscription_id=sub_id, data_name="nodes_data")
        relations_data = self.load_knowledge_graph(subscription_id=sub_id, data_name="relations_data")
        records_data = self.load_knowledge_graph(subscription_id=sub_id, data_name="records_data")
        term_mapping = self.load_knowledge_graph(subscription_id=sub_id, data_name="term_mapping")
        
        knowledge_data = [
            {'metadata': {'nodes': json.loads(nodes_data)}},
            {'metadata': {'relations': json.loads(relations_data)}},
            {'metadata': {'records': json.loads(records_data)}},
            {'metadata': {'terms': json.loads(term_mapping)}}
        ]
        
        return {
            "embedding": embedding,
            "faiss": f_data,
            "knowledge_graph": knowledge_data
        }
    
    def _gzip_compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=6)  

    def _sha256(self, data: bytes) -> bytes:
        return hashlib.sha256(data).digest()
    
    def _json_minify(self, obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

    def _wrap_compressed_json(self, text: str) -> str:
        raw = text.encode("utf-8")
        comp = zlib.compress(raw) 
        b64 = base64.b64encode(comp).decode("ascii")
        wrapper = {
            "__compressed__": True,
            "codec": "zlib",
            "encoding": "base64",
            "orig_len": len(raw),
            "compressed_len": len(comp),
            "payload": b64,
        }
        return self._json_minify(wrapper)
    
    def _maybe_compress_metadata(self, metadata: Optional[Dict]) -> str:
        if metadata is None:
            metadata = {}
        text = self._json_minify(metadata)
        if len(text.encode("utf-8")) < self.METADATA_COMPRESS_THRESHOLD:
            return text
        return self._wrap_compressed_json(text)
    
    def unpack_metadata(self, metadata_text: Optional[str]) -> Dict:
        if not metadata_text:
            return {}
        try:
            obj = json.loads(metadata_text)
        except Exception:
            return {}

        if not isinstance(obj, dict) or not obj.get("__compressed__"):
            return obj if isinstance(obj, dict) else {}

        if obj.get("codec") != "zlib" or obj.get("encoding") != "base64":
            logger.warning("Unknown metadata compression format")
            return {}

        try:
            comp = base64.b64decode(obj.get("payload") or "")
            raw = zlib.decompress(comp)
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.warning(f"metadata decompress failed: {e}")
            return {}
    
    def count_prompts(self, sub_id=None, filter=None) -> int:
        try:
            sql = self.queries.get('COUNT_SQL') + " WHERE 1=1"
            params = []

            if sub_id:
                sql += " AND p.subscription_id = %s"
                params.append(sub_id)
            else:
                sql += " AND p.subscription_id IS NULL"

            if filter and filter.get("keywords"):
                for kw in filter["keywords"]:
                    sql += " AND (p.name LIKE %s OR p.description LIKE %s OR p.type = %s)"
                    kw_like = f"%{kw}%"
                    params.extend([kw_like, kw_like, kw])

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, tuple(params))
                    row = cursor.fetchone()
                    return row["cnt"]
        except Exception as e:
            logger.error(f"count_prompts: {e}")
            return None
        
    def get_prompts(self, s_id=None, model=None, filter=None, tiny=False):
        try:
            if model:
                sub_id = [s_id]
                sub_id.append(1)
            else:
                sub_id = s_id
                
            sql = self.queries.get('PROMPT_TINY_SQL') if tiny else self.queries.get('PROMPT_SQL')
            sql += " WHERE 1=1"
            params = []

            if sub_id is not None:
                if isinstance(sub_id, (list, tuple, set)): 
                    placeholders = ",".join(["%s"] * len(sub_id))
                    sql += f" AND p.subscription_id IN ({placeholders})"
                    params.extend(sub_id)
                else:  
                    sql += " AND p.subscription_id = %s"
                    params.append(sub_id)
            else:
                sql += " AND p.subscription_id IS NULL"
            
            if filter and filter.get("keywords"):
                for kw in filter["keywords"]:
                    sql += " AND (p.name LIKE %s OR p.description LIKE %s OR p.type = %s)"
                    kw_like = f"%{kw}%"
                    params.extend([kw_like, kw_like, kw])

            if filter:
                sql += " LIMIT %s OFFSET %s"
                params.extend([filter.get("limit", 50), filter.get("offset", 0)])

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, tuple(params))
                    rows = cursor.fetchall()
                    for row in rows:
                        row["tiny"] = tiny
                        
                        if "source" in row and isinstance(row["source"], str):
                            if row["source"].startswith("gpt/"):
                                row["source"] = row["source"].removeprefix("gpt/")
                            elif row["source"].startswith("llama/"):
                                row["source"] = row["source"].removeprefix("llama/")
                        
                    logger.info(f"임베딩 로드 완료")
                    return rows
        except Exception as e:
            logger.error(f"임베딩 로드 실패(get_prompts): {e}")
            return None
        
    def get_prompt_by_name(self, sub_id, name):
        try:
            sql = self.queries.get('PROMPT_SQL') + " WHERE p.name = %s"
            params = [name]

            if sub_id is not None:
                sql += " AND p.subscription_id = %s"
                params.append(sub_id)
            else:
                sql += " AND p.subscription_id IS NULL"
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, tuple(params))
                    row = cursor.fetchone()
                    
                    return row
        except Exception as e:
            logger.error(f"임베딩 로드 실패(get_prompt_by_name): {e}")
            return None
    
    def get_prompt_by_guid(self, sub_id, guid):
        try:
            sql = self.queries.get('PROMPT_SQL') + " WHERE p.guid = %s"
            params = [guid]

            if sub_id is not None:
                sql += " AND p.subscription_id = %s"
                params.append(sub_id)
            else:
                sql += " AND p.subscription_id IS NULL"
                
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, tuple(params))
                    return cursor.fetchone()
        except Exception as e:
            logger.error(f"임베딩 로드 실패(get_prompt_by_guid): {e}")
            return None
        
    def get_prompts_by_guids(self, sub_id, guids):
        try:
            if not guids:
                return []

            placeholders = ",".join(["%s"] * len(guids))
            sql = self.queries.get('PROMPT_SQL') + f" WHERE p.guid IN ({placeholders})"
            params = guids

            if sub_id:
                sql += " AND p.subscription_id = %s"
                params.append(sub_id)

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(sql, tuple(params))
                    return cursor.fetchall()
        except Exception as e:
            logger.error(f"임베딩 로드 실패(get_prompt_by_guids): {e}")
            return None
        
    def get_prompt_by_guid_or_name(self, sub_id, prompt: OpenAiPrompt):
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    where_clauses = []
                    params = []

                    or_conditions = []
                    if prompt.guid:
                        or_conditions.append("p.guid = %s")
                        params.append(str(prompt.guid))
                    if prompt.name:
                        or_conditions.append("p.name = %s")
                        params.append(prompt.name)

                    if not or_conditions:
                        return None  

                    where_clauses.append(f"({' OR '.join(or_conditions)})")

                    if sub_id:
                        where_clauses.append("p.subscription_id = %s")
                        params.append(sub_id)
                    else:
                        where_clauses.append("p.subscription_id IS NULL")

                    sql = self.queries.get('PROMPT_SQL') + " WHERE " + " AND ".join(where_clauses)
                    cursor.execute(sql, params)
                    row = cursor.fetchone()

                    if not row:
                        return None

                    return row          
        except Exception as e:
            logger.error(f"임베딩 로드 실패(get_prompt_by_guid_or_name): {e}")
            return None
        
    def get_vector_checksums(self, s_id, model=None):

        if model:
            sub_id = [s_id]
            sub_id.append(1)
        else:
            if isinstance(s_id, list):
                sub_id = s_id
            else:
                sub_id = [s_id]

        placeholders = ','.join(['%s'] * len(sub_id))
        result = []
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = f"""
                        SELECT
                            source,
                            MD5(CONVERT(prompt USING utf8mb4)) AS vector_hash,
                            token_count,
                            UNIX_TIMESTAMP(created_at) AS created_ts,
                            UNIX_TIMESTAMP(updated_at) AS updated_ts
                        FROM openai_prompts
                        WHERE subscription_id IN ({placeholders})
                    """
                    cursor.execute(sql, sub_id)
                    result = cursor.fetchall()

        except Exception as e:
            logger.error(f"벡터 체크섬 로드 실패(get_vector_checksums): {e}")
            return None

        return result
    
    def insert_prompt(self, sub_id, prompt: OpenAiPrompt, model, source, embedding_vector, tags=None) -> Optional[int]:
        found_prompt = self.get_prompt_by_guid_or_name(sub_id, prompt)
        try:
            if found_prompt:
                if prompt.name == found_prompt['name']:
                    raise ValueError(f"Duplicated prompt name: {prompt.name}")
                raise ValueError(f"Duplicated prompt guid: {prompt.guid}")
        except Exception as e:
            logger.error(f"임베딩 저장 실패: {e}")
            return None
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    guid = uuid.uuid4()
                    
                    cursor.execute(
                        self.queries.get('INSERT_PROMPT'),
                        (guid,
                         sub_id,
                         prompt.name,
                         prompt.description,
                         prompt.type,
                         prompt.prompt,
                         len(embedding_vector),
                         embedding_vector.tobytes(),
                         source,
                         tags)
                    )
                    conn.commit()
                    embedding_id = cursor.lastrowid
                    
                    logger.info(f"임베딩 저장 완료: ID={embedding_id}, Doc={guid}")
                    return embedding_id
        except Exception as e:
            conn.rollback()
            logger.error(f"임베딩 저장 실패: {e}")
            return None
        
    def insert_prompt_values(self, name, description, prompt_type, prompt_data, model, source, embedding_vector, tags=None) -> Optional[int]:
        prompt = OpenAiPrompt(
            name=name,
            description=description,
            type=prompt_type,
            prompt=prompt_data
        )
        
        if model == "llama":
            sub_id = 0
        elif model == "gpt":
            sub_id = 1
        else:
            sub_id = None
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    guid = uuid.uuid4()
                    
                    cursor.execute(
                        self.queries.get('INSERT_PROMPT'),
                        (guid,
                         sub_id,
                         prompt.name,
                         prompt.description,
                         prompt.type,
                         prompt.prompt,
                         len(embedding_vector),
                         embedding_vector.tobytes(),
                         source,
                         tags)
                    )
                    embedding_id = cursor.lastrowid
                conn.commit()
                       
            return embedding_id
        except Exception as e:
            conn.rollback()
            logger.error(f"임베딩 저장 실패: {e}")
            return None
    
    def batch_insert_prompts(self, batch_data, batch_size = 500):
        if not batch_data:
            return 0

        params_buffer = []
        for item in batch_data:
            name = item.get("name")
            description = item.get("description")
            prompt_type = item.get("prompt_type")
            prompt_data = item.get("prompt_data")
            model = item.get("model")
            source = item.get("source")
            token_count = item.get("token_count")
            embedding_vector = item.get("embedding_vector")
            question_vector = item.get("question_vector")
            tags = item.get("tags")
            guid = item.get("guid")

            if not guid:
                guid = uuid.uuid4()
            sub_id = item.get("sub_id")

            params_buffer.append((
                guid,
                sub_id,
                name,
                description,
                prompt_type,
                prompt_data,
                token_count,
                embedding_vector,
                question_vector,
                source,
                tags
            ))

        total_inserted = 0
        try:

            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    for i in range(0, len(params_buffer), batch_size):
                        chunk = params_buffer[i:i + batch_size]
                        cursor.executemany(self.queries.get('INSERT_PROMPT'), chunk)
                        total_inserted += cursor.rowcount
                conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            raise

        return total_inserted
        
    def update_prompt(self, sub_id, guid, prompt: OpenAiPrompt, embedding_vector) -> Optional[int]:
        try:
            old = self.get_prompt_by_guid(sub_id, guid)
            if old is None:
                raise RuntimeError(f"prompt not found: {guid}")
        except Exception as e:
            logger.error(f"임베딩 수정 실패: {e}")
            return None
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    if old['prompt'] is None or (old['prompt'] == prompt.prompt and old['vector'] is not None):
                        prompt.token_count = old['token_count']

                        sql = self.queries.get('UPDATE_PROMPT')
                        cursor.execute(sql, (
                            prompt.name,
                            prompt.description,
                            prompt.type,
                            prompt.prompt,
                            guid
                        ))
                        conn.commit()
                    else:
                        sql = self.queries.get('UPDATE_PROMPT_VECTOR')
                        cursor.execute(sql, (
                            prompt.name,
                            prompt.description,
                            prompt.type,
                            prompt.prompt,
                            len(embedding_vector),
                            embedding_vector.tobytes(),
                            guid
                        ))
                    embedding_id = cursor.lastrowid
                conn.commit() 
            return embedding_id
        except Exception as e:
            conn.rollback()
            logger.error(f"임베딩 수정 실패: {e}")
            return None
    
    def update_prompt_values(self, old, sub_id, guid, name, description, prompt_type, prompt_data, embedding_vector, source=None, tags=None) -> Optional[int]:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cond_no_vec = (
                        old.get("prompt") is None
                        or (old.get("prompt") == prompt_data and old.get("vector") is not None)
                    )
                    if cond_no_vec:
                        sql = self.queries.get('UPDATE_PROMPT')
                        cursor.execute(sql, (
                            name,
                            description,
                            prompt_type,
                            prompt_data,
                            guid
                        ))
                        conn.commit()
                    else:
                        sql = self.queries.get('UPDATE_PROMPT_VECTOR')
                        cursor.execute(sql, (
                            name,
                            description,
                            prompt_type,
                            prompt_data,
                            len(embedding_vector),
                            embedding_vector.tobytes(),
                            source,
                            tags,
                            guid
                        ))
                    embedding_id = cursor.lastrowid
                conn.commit() 
            return embedding_id
        except Exception as e:
            conn.rollback()
            logger.error(f"임베딩 수정 실패: {e}")
            return None
    
    def batch_update_prompts(self, batch_data, batch_size = 500):
        if not batch_data:
            return 0

        params_with_vec = [] 

        for item in batch_data:
            old = item.get("old") or {}
            guid = item.get("guid")
            name = item.get("name")
            description = item.get("description")
            prompt_type = item.get("prompt_type")
            prompt_data = item.get("prompt_data")
            token_count = item.get("token_count")
            embedding_vector = item.get("embedding_vector")
            question_vector = item.get("question_vector")
            source = item.get("source")
            tags = item.get("tags")

            params_with_vec.append((
                name,
                description,
                prompt_type,
                prompt_data,
                token_count,
                embedding_vector,
                question_vector,
                source,
                tags,
                guid
            ))

        total_updated = 0
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql_with_vec = self.queries.get('UPDATE_PROMPT_VECTOR')  
                    for i in range(0, len(params_with_vec), batch_size):
                        chunk = params_with_vec[i:i + batch_size]
                        cursor.executemany(sql_with_vec, chunk)
                        total_updated += cursor.rowcount

                conn.commit()
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            raise

        return total_updated
    
    def remove_prompt(self, sub_id, guids):
        try:
            olds = self.get_prompts_by_guids(sub_id, guids)
            
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    placeholders = ','.join(['%s'] * len(guids))
                    sql = f"DELETE FROM openai_prompts WHERE guid IN ({placeholders})"

                    params = list(str(guid) for guid in guids)

                    if sub_id is not None:
                        sql += " AND subscription_id = %s"
                        params.append(sub_id)

                    cursor.execute(sql, params)
                    conn.commit()
        except Exception as e:
            raise RuntimeError(f"cannot remove prompts {guids}") from e
    
    def get_embeddings_by_document(self, doc_id: int) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        self.queries.get('GET_EMBEDDINGS_BY_DOCUMENT'),
                        (doc_id,)
                    )
                    results = cursor.fetchall()
                    
                    for result in results:
                        if result.get('embedding_vector'):
                            vector_bytes = base64.b64decode(result['embedding_vector'])
                            result['embedding_vector'] = pickle.loads(vector_bytes)
                    
                    logger.info(f"임베딩 조회 완료: Doc={doc_id}, Count={len(results)}")
                    return results
        except Exception as e:
            logger.error(f"임베딩 조회 실패: {e}")
            return []
    
    def search_similar_embeddings(self, query_vector: np.ndarray,
                                 top_k: int = 5) -> List[Dict]:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(self.queries.get('GET_ALL_EMBEDDINGS'))
                    all_embeddings = cursor.fetchall()
                    
                    if not all_embeddings:
                        return []
                    
                    similarities = []
                    for emb in all_embeddings:
                        vector_bytes = base64.b64decode(emb['embedding_vector'])
                        stored_vector = pickle.loads(vector_bytes)
                        
                        similarity = np.dot(query_vector, stored_vector) / (
                            np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
                        )
                        
                        similarities.append({
                            'embedding_id': emb['embedding_id'],
                            'doc_id': emb['doc_id'],
                            'chunk_text': emb['chunk_text'],
                            'similarity': float(similarity)
                        })
                    
                    similarities.sort(key=lambda x: x['similarity'], reverse=True)
                    results = similarities[:top_k]
                    
                    logger.info(f"유사 임베딩 검색 완료: Top {len(results)}")
                    return results
        except Exception as e:
            logger.error(f"유사 임베딩 검색 실패: {e}")
            return []
        
    def toggle_prompt(self, user_info, guid, enabled: bool):
        prompt = self.get_prompt_by_guid(user_info, guid)
        if prompt is None:
            return

        if prompt['enabled'] == enabled:
            return

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    sql = "UPDATE openai_prompts SET enabled=%s WHERE guid=%s"
                    cursor.execute(sql, (enabled, guid))
                    
        except Exception as e:
            logger.error(f"프롬프트 토글 실패: {e}")
            return []
    
    def save_faiss_index(self, sub_id, index_name: str, index_data: bytes,
                     metadata: Optional[Dict] = None) -> Optional[int]:
        try:
            meta_json = json.dumps(metadata or {}, ensure_ascii=False)
            packed_metadata = self._maybe_compress_metadata(metadata)

            with self.get_connection() as conn:
                conn.ping(reconnect=True)
                ac = conn.get_autocommit()
                if ac: conn.autocommit(False)
                try:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            self.queries.get('GET_FAISS_INDEX_BY_NAME'),
                            (sub_id, index_name)
                        )
                        row = cursor.fetchone()
                        if row:
                            cursor.execute(
                                self.queries.get('UPDATE_FAISS_INDEX'),
                                (b"", meta_json, packed_metadata, sub_id, index_name) 
                            )
                        else:
                            cursor.execute(
                                self.queries.get('INSERT_FAISS_INDEX'),
                                (sub_id, index_name, b"", meta_json, packed_metadata)
                            )

                        cursor.execute(
                            self.queries.get('DELETE_FAISS_PARTS'),
                            (sub_id, index_name)
                        )

                        mv = memoryview(index_data)
                        total = len(mv)
                        part_no = 0
                        offset = 0
                        ins_sql = self.queries.get('INSERT_FAISS_PARTS')
                        while offset < total:
                            chunk = bytes(mv[offset:offset + self.CHUNK_SIZE])
                            cursor.execute(ins_sql, (sub_id, index_name, part_no, chunk))
                            part_no += 1
                            offset += len(chunk)

                        cursor.execute(
                            self.queries.get('GET_FAISS_INDEX_BY_NAME'),
                            (sub_id, index_name)
                        )
                        row2 = cursor.fetchone()
                        if not row2:
                            raise RuntimeError("faiss row not found after save")
                        index_id = row2['id'] if isinstance(row2, dict) else row2[0]

                    conn.commit()
                    return index_id
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    if ac: conn.autocommit(True)
        except Exception as e:
            logger.exception("FAISS 인덱스 저장 실패: %s", e)
            return None
    
    def load_faiss_index(self, sub_id, index_name: str, stream: bool = False) -> Optional[Tuple[bytes, Dict]]:
        try:
            with self.get_connection() as conn:
                conn.ping(reconnect=True)
                with conn.cursor() as cursor:
                    cursor.execute(
                        self.queries.get('GET_FAISS_INDEX_BY_NAME'),
                        (sub_id, index_name)
                    )
                    row = cursor.fetchone()
                    if not row:
                        logger.warning("FAISS 인덱스 없음: %s", index_name)
                        return None

                    if isinstance(row, dict):
                        meta_text      = row.get('metadata') or "{}"
                        artifact_text  = row.get('artifact_metadata') or "{}"
                        legacy_blob    = row.get('index_data') or b""
                    else:
                        meta_text      = row[4] if len(row) > 4 else "{}"
                        artifact_text  = row[5] if len(row) > 5 else "{}"
                        legacy_blob    = row[3] if len(row) > 3 else b""

                    metadata  = self.unpack_metadata(meta_text) if meta_text else {}
                    try:
                        artifact = (self.unpack_metadata(artifact_text) if artifact_text else {}) \
                                if hasattr(self, "unpack_metadata") else (json.loads(artifact_text) if artifact_text else {})
                    except Exception:
                        artifact = {}

                    if artifact:
                        metadata = dict(metadata) 
                        metadata.setdefault("_artifact", artifact)

                    if not stream:
                        cursor.execute(
                            self.queries.get('GET_FAISS_PARTS'),
                            (sub_id, index_name)
                        )
                        parts = cursor.fetchall()
                        if not parts:
                            return legacy_blob, metadata

                        buf = bytearray()
                        for p in parts:
                            chunk = (p['chunk'] if isinstance(p, dict) else p[1]) if p is not None else None
                            if chunk:
                                buf.extend(chunk)
                        return bytes(buf), metadata

                    cursor.execute(
                        self.queries.get('GET_FAISS_PARTS'),
                        (sub_id, index_name)
                    )
                    buf = bytearray()
                    while True:
                        batch = cursor.fetchmany(size=256)  
                        if not batch:
                            break
                        for p in batch:
                            chunk = (p['chunk'] if isinstance(p, dict) else p[1]) if p is not None else None
                            if chunk:
                                buf.extend(chunk)
                    if not buf:
                        return legacy_blob, metadata

                    return bytes(buf), metadata

        except Exception as e:
            logger.error("FAISS 인덱스 로드 실패: %s", e)
            return None
    
    def load_all_faiss_index(self, sub_id, use_parts: bool = True, stream_parts: bool = False) -> Optional[List[Dict]]:
        try:
            with self.get_connection() as conn:
                conn.ping(reconnect=True)
                with conn.cursor() as cursor:
                    cursor.execute(
                        self.queries.get('GET_ALL_FAISS_INDEXES'),
                        (sub_id,)
                    )
                    rows = cursor.fetchall()
                    if not rows:
                        logger.warning("FAISS 데이터 없음")
                        return None

                    results: List[Dict] = []
                    for row in rows:
                        if isinstance(row, dict):
                            _id         = row["id"]
                            _sub_id     = row["subscription_id"]
                            _name       = row["index_name"]
                            _blob_main  = row.get("index_data") or b""
                            _meta_text  = row.get("metadata") or "{}"
                            _artifact_text = row.get("artifact_metadata") or "{}"
                            _intent     = row.get("intent")
                            _created_at = row["created_at"]
                            _updated_at = row["updated_at"]
                        else:
                            _id         = row[0]
                            _sub_id     = row[1]
                            _name       = row[2]
                            _blob_main  = row[3] if len(row) > 3 else b""
                            _meta_text  = row[4] if len(row) > 4 else "{}"
                            _artifact_text = row[5] if len(row) > 5 else "{}"
                            _intent     = row[6] if len(row) > 6 else None
                            _created_at = row[7] if len(row) > 7 else None
                            _updated_at = row[8] if len(row) > 8 else None

                        _metadata = self.unpack_metadata(_meta_text) if _meta_text else {}
                        try:
                            _artifact = (self.unpack_metadata(_artifact_text) if _artifact_text else {}) \
                                        if hasattr(self, "unpack_metadata") else (json.loads(_artifact_text) if _artifact_text else {})
                        except Exception:
                            _artifact = {}

                        _blob = _blob_main or b""
                        if use_parts:
                            loaded = self.load_faiss_index(_sub_id, _name, stream=stream_parts)
                            if loaded is not None:
                                _blob_from_parts, _ = loaded  
                                if _blob_from_parts:
                                    _blob = _blob_from_parts

                        results.append({
                            "id": _id,
                            "subscription_id": _sub_id,
                            "index_name": _name,
                            "index_data": _blob,
                            "metadata": _metadata,                 
                            "artifact_metadata": _artifact,       
                            "intent": _intent,                     
                            "created_at": _created_at,
                            "updated_at": _updated_at,
                        })

                    logger.info("FAISS 인덱스 로드 완료: %d개", len(results))
                    return results

        except Exception as e:
            logger.error("FAISS 인덱스 로드 실패: %s", e)
            return None
    
    def delete_faiss_index(self, subscription_id) -> bool:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        self.queries.get('DELETE_FAISS_INDEX'),
                        (subscription_id,)
                    )
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        logger.info(f"FAISS 인덱스 삭제 완료: {subscription_id}")
                        return True
                    return False
        except Exception as e:
            logger.error(f"FAISS 인덱스 삭제 실패: {e}")
            return False
    
    def save_knowledge_graph(self,
                         subscription_id: Optional[int],
                         data_name: str,
                         json_text: str) -> int:
        
        raw = json_text.encode('utf-8')
        orig_size = len(raw)
        compressed = self._gzip_compress(raw)
        digest_all = self._sha256(compressed)
        
        parts: List[Tuple[int, bytes, bytes]] = []
        total_parts = math.ceil(len(compressed) / self.CHUNK_SIZE)
        for i in range(total_parts):
            chunk = compressed[i*self.CHUNK_SIZE:(i+1)* self.CHUNK_SIZE]
            parts.append((i, chunk, self._sha256(chunk)))

        pointer_json = {
            "storage": "parts",
            "codec": "gzip",
            "orig_size": orig_size,
            "parts": total_parts,
            "sha256": hashlib.sha256(compressed).hexdigest()
        }

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                conn.begin()

                sql_upsert = self.queries.get('INSERT_KG_CHUNK')
                cur.execute(sql_upsert, (subscription_id, data_name, pointer_json["orig_size"], pointer_json["parts"], pointer_json["sha256"]))

                sql_get_id = self.queries.get('GET_KG_ID')
                cur.execute(sql_get_id, (subscription_id, data_name))
                row = cur.fetchone()
                if not row:
                    conn.rollback()
                    raise RuntimeError("kg_id fetch failed")
                
                kg_id = row['id'] if isinstance(row, dict) else row[0]

                cur.execute(self.queries.get('DEL_CHUNK_DATA'), (kg_id,))

                sql_ins_part = self.queries.get('INSERT_CHUNK_DATA')
                cur.executemany(sql_ins_part, [(kg_id, i, chunk, chk) for i, chunk, chk in parts])

                conn.commit()
                return kg_id

    def load_knowledge_graph(self, subscription_id: Optional[int], data_name: str) -> str:
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(self.queries.get('GET_METADATA_INFO'), (subscription_id, data_name))
                row = cur.fetchone()
                if not row:
                    raise KeyError("graph not found")
                kg_id = row['id']
                
                cur.execute(self.queries.get('GET_METADATA'), (kg_id,))
                parts = cur.fetchall()

                if not parts:
                    raise RuntimeError("no parts stored")

                buf = b''.join([p['chunk'] for p in parts])  

                decompressed = zlib.decompress(buf)
                return decompressed.decode('utf-8')
    
    def delete_knowledge_graph(self, subscription_id) -> bool:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        self.queries.get('DELETE_KG_ID'),
                        (subscription_id,)
                    )
                    conn.commit()
                    
                    if cursor.rowcount > 0:
                        logger.info(f"지식그래프 삭제 완료: {subscription_id}")
                        return True
                    return False
        except Exception as e:
            logger.error(f"지식그래프 삭제 실패: {e}")
            return False

    
    def get_statistics(self) -> Dict[str, Any]:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    stats = {}
                    
                    cursor.execute("SELECT COUNT(*) as count FROM documents")
                    stats['total_documents'] = cursor.fetchone()['count']
                    
                    cursor.execute("SELECT COUNT(*) as count FROM embeddings")
                    stats['total_embeddings'] = cursor.fetchone()['count']
                    
                    cursor.execute("SELECT COUNT(*) as count FROM faiss_indexes")
                    stats['total_faiss_indexes'] = cursor.fetchone()['count']
                    
                    cursor.execute(
                        "SELECT AVG(vector_dimension) as avg_dim FROM embeddings"
                    )
                    result = cursor.fetchone()
                    stats['avg_embedding_dimension'] = float(result['avg_dim'] or 0)
                    
                    logger.info("통계 정보 조회 완료")
                    return stats
        except Exception as e:
            logger.error(f"통계 정보 조회 실패: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30) -> bool:
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        self.queries.get('DELETE_OLD_DOCUMENTS'),
                        (days,)
                    )
                    deleted_docs = cursor.rowcount
                    
                    cursor.execute(
                        self.queries.get('DELETE_ORPHAN_EMBEDDINGS')
                    )
                    deleted_embs = cursor.rowcount
                    
                    conn.commit()
                    
                    logger.info(f"데이터 정리 완료: 문서 {deleted_docs}건, 임베딩 {deleted_embs}건")
                    return True
        except Exception as e:
            logger.error(f"데이터 정리 실패: {e}")
            return False

    def get_single_embedding(self, subscription_id: str, source: str) -> Optional[Dict]:
        """
        특정 파일의 임베딩 정보 조회
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    query = """
                        SELECT id, source,
                               MD5(CONVERT(prompt USING utf8mb4)) AS checksum,
                               token_count,
                               LENGTH(vector) as vector_size,
                               created_at, updated_at,
                               prompt
                        FROM openai_prompts
                        WHERE subscription_id = %s AND source = %s
                    """
                    cursor.execute(query, (subscription_id, source))
                    result = cursor.fetchone()

                    if result:
                        # 벡터 차원 계산 (float32 = 4 bytes)
                        result['vector_dim'] = result['vector_size'] // 4 if result['vector_size'] else 0
                        return result
                    return None
        except Exception as e:
            logger.error(f"임베딩 조회 실패: {e}")
            return None

    def update_single_embedding(self, subscription_id: str, source: str,
                              embedding, content: str, checksum: str) -> bool:
        """
        특정 파일의 임베딩 업데이트
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 벡터를 바이너리로 변환
                    vector_binary = np.array(embedding.vector, dtype=np.float32).tobytes()

                    query = """
                        UPDATE openai_prompts
                        SET vector = %s,
                            token_count = %s,
                            prompt = %s,
                            updated_at = NOW()
                        WHERE subscription_id = %s AND source = %s
                    """

                    cursor.execute(query, (
                        vector_binary,
                        embedding.token_count,
                        content,
                        subscription_id,
                        source
                    ))

                    if cursor.rowcount > 0:
                        conn.commit()
                        logger.info(f"임베딩 업데이트 성공: {source}")
                        return True
                    else:
                        logger.warning(f"업데이트할 임베딩 없음: {source}")
                        return False

        except Exception as e:
            logger.error(f"임베딩 업데이트 실패: {e}")
            return False

    def insert_single_embedding(self, subscription_id: str, source: str,
                              embedding, content: str, checksum: str) -> bool:
        """
        새 임베딩 삽입
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # 벡터를 바이너리로 변환
                    vector_binary = np.array(embedding.vector, dtype=np.float32).tobytes()

                    # GUID 생성
                    guid = str(uuid.uuid4())

                    query = """
                        INSERT INTO openai_prompts
                        (guid, subscription_id, name, description, enabled, type,
                         prompt, token_count, vector, source,
                         created_at, updated_at)
                        VALUES (%s, %s, %s, %s, 1, 'embedding',
                                %s, %s, %s, %s,
                                NOW(), NOW())
                    """

                    cursor.execute(query, (
                        guid,
                        subscription_id,
                        source,  # name으로 사용
                        f"Embedding for {source}",  # description
                        content,  # prompt 필드에 내용 저장
                        embedding.token_count,
                        vector_binary,
                        source
                    ))

                    conn.commit()
                    logger.info(f"임베딩 삽입 성공: {source}")
                    return True

        except Exception as e:
            logger.error(f"임베딩 삽입 실패: {e}")
            return False

    def get_active_subscription_ids(self) -> List[str]:
        """
        만료되지 않은 모든 활성 구독 ID를 가져옴

        Returns:
            활성 구독 ID 리스트
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # expires_at이 현재 시간보다 미래인 구독키만 조회
                    sql = """
                        SELECT id
                        FROM ai_subscriptions
                        WHERE expires_at > NOW()
                        ORDER BY id
                    """
                    cursor.execute(sql)
                    result = cursor.fetchall()

                    # DictCursor를 사용하므로 'id' 키로 접근
                    subscription_ids = [str(row['id']) for row in result if row.get('id')]

                    logger.info(f"활성 구독키 {len(subscription_ids)}개 조회: {', '.join(subscription_ids)}")
                    return subscription_ids

        except Exception as e:
            logger.error(f"구독키 조회 실패: {e}")
            import traceback
            traceback.print_exc()
            # 오류 시 기본값 반환
            return ["1"]