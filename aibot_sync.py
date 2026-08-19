"""
프롬프트 임베딩 싱크 모듈

메인 서버의 Qdrant(sub_id=1) 데이터와 로컬 Qdrant를 동기화한다.
- 부팅 시 1회 실행
- 이후 주기적으로 실행 (config [sync] interval_seconds)

싱크 흐름:
  1. 메인서버 GET /api/ai/sync/manifest → 경량 목록 (guid, rev, content_hash, enabled, doc_type)
  2. 로컬 Qdrant scroll(sub_id=1) → 동일 형태 목록
  3. diff 계산 (추가/삭제/업데이트/토글)
  4. 변경분만 메인서버 POST /api/ai/sync/points 로 벡터 포함 전체 데이터 요청
  5. 로컬 Qdrant에 upsert/delete/toggle 반영
  6. RAG 시스템 리로드
"""

import asyncio
import hashlib
import logging
import time
from typing import Dict, List, Optional, Any

import requests
from qdrant_client import models

from config_utils import ConfigManager, qdrant_collection

logger = logging.getLogger(__name__)

SYNC_SUB_ID = 1
COLLECTION_NAME = qdrant_collection()
MANIFEST_BATCH_SIZE = 256  # scroll 배치 크기
POINTS_REQUEST_BATCH = 50  # 한 번에 요청할 guid 수


class SyncClient:
    """메인 서버와 로컬 Qdrant 간 싱크 클라이언트"""

    def __init__(self, config: ConfigManager, qdrant_client, llm_handler=None):
        self._config = config
        self._qdrant = qdrant_client
        self._llm_handler = llm_handler

        # sync 설정 로드
        self._enabled = config.config.get('sync', 'enabled', fallback='False').lower() == 'true'
        self._main_server_url = config.config.get('sync', 'main_server_url', fallback='').rstrip('/')
        self._api_key = config.config.get('sync', 'api_key', fallback='')
        self._interval = int(config.config.get('sync', 'interval_seconds', fallback='300'))

    @property
    def enabled(self) -> bool:
        return self._enabled and bool(self._main_server_url)

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    # ──────────────────────────────────────────────
    # 1. 메인 서버에서 manifest 가져오기
    # ──────────────────────────────────────────────
    def fetch_remote_manifest(self) -> Dict[str, Dict]:
        """메인 서버의 sub_id=1 경량 목록 조회

        Returns:
            {guid: {rev, content_hash, enabled, doc_type}, ...}
        """
        url = f"{self._main_server_url}/api/ai/sync/manifest"
        resp = requests.get(url, headers=self._headers(), timeout=30)
        resp.raise_for_status()
        data = resp.json()

        remote = {}
        for item in data.get("items", []):
            remote[item["guid"]] = {
                "rev": item.get("rev", 0),
                "content_hash": item.get("content_hash", ""),
                "enabled": item.get("enabled", True),
                "doc_type": item.get("doc_type", "qna"),
            }
        return remote

    # ──────────────────────────────────────────────
    # 2. 로컬 Qdrant에서 manifest 수집
    # ──────────────────────────────────────────────
    def collect_local_manifest(self) -> Dict[str, Dict]:
        """로컬 Qdrant sub_id=1 포인트들의 경량 정보 수집

        Returns:
            {guid: {rev, content_hash, enabled, doc_type, point_id}, ...}
        """
        local = {}
        offset = None

        while True:
            result = self._qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="sub_id",
                            match=models.MatchValue(value=SYNC_SUB_ID)
                        ),
                    ]
                ),
                with_payload=True,
                with_vectors=False,
                limit=MANIFEST_BATCH_SIZE,
                offset=offset,
            )

            points, next_offset = result
            for pt in points:
                p = pt.payload or {}
                guid = p.get("guid")
                if not guid:
                    continue
                local[guid] = {
                    "rev": p.get("rev", 0),
                    "content_hash": p.get("content_hash", ""),
                    "enabled": p.get("enabled", True),
                    "doc_type": p.get("doc_type", "qna"),
                    "point_id": pt.id,
                }

            if next_offset is None:
                break
            offset = next_offset

        return local

    # ──────────────────────────────────────────────
    # 3. diff 계산
    # ──────────────────────────────────────────────
    def compute_diff(self, remote: Dict, local: Dict) -> Dict[str, List[str]]:
        """remote vs local 비교하여 작업 목록 산출

        Returns:
            {
                "to_create": [guid, ...],    # 원격에만 있음
                "to_delete": [guid, ...],    # 로컬에만 있음
                "to_update": [guid, ...],    # content_hash 다름
                "to_toggle": [guid, ...],    # content_hash 같지만 enabled 다름
            }
        """
        remote_guids = set(remote.keys())
        local_guids = set(local.keys())

        to_create = list(remote_guids - local_guids)
        # 로컬에만 있는 데이터는 삭제하지 않음 (로컬 임의 추가분 보존)
        to_update = []
        to_toggle = []

        for guid in remote_guids & local_guids:
            r, l = remote[guid], local[guid]
            if r["content_hash"] != l["content_hash"]:
                to_update.append(guid)
            elif r["enabled"] != l["enabled"]:
                to_toggle.append(guid)

        return {
            "to_create": to_create,
            "to_delete": [],
            "to_update": to_update,
            "to_toggle": to_toggle,
        }

    # ──────────────────────────────────────────────
    # 4. 메인 서버에서 포인트 전체 데이터 가져오기
    # ──────────────────────────────────────────────
    def fetch_remote_points(self, guids: List[str]) -> List[Dict]:
        """메인 서버에서 guid 목록에 해당하는 벡터 + payload 전체 데이터 조회

        Returns:
            [{id, vectors: {dense, sparse, colbert}, payload: {...}}, ...]
        """
        if not guids:
            return []

        url = f"{self._main_server_url}/api/ai/sync/points"
        headers = self._headers()
        headers["Accept-Encoding"] = "gzip"  # gzip 압축 수신 요청
        all_points = []

        # 배치 단위로 요청 (한 번에 너무 많은 데이터 방지)
        for i in range(0, len(guids), POINTS_REQUEST_BATCH):
            batch_guids = guids[i:i + POINTS_REQUEST_BATCH]
            resp = requests.post(
                url,
                json={"guids": batch_guids},
                headers=headers,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            all_points.extend(data.get("points", []))
            logger.info(f"[Sync] points 수신: {i + len(batch_guids)}/{len(guids)}")

        return all_points

    # ──────────────────────────────────────────────
    # 5. 로컬 Qdrant 반영
    # ──────────────────────────────────────────────
    def _build_point_struct(self, pt: Dict) -> Optional[models.PointStruct]:
        """원격 포인트 dict → Qdrant PointStruct 변환"""
        try:
            vectors = pt["vectors"]
            named_vectors = {}

            if "dense" in vectors:
                named_vectors["dense"] = vectors["dense"]
            if "colbert" in vectors and vectors["colbert"]:
                named_vectors["colbert"] = vectors["colbert"]
            if "sparse" in vectors and vectors["sparse"]:
                sp = vectors["sparse"]
                named_vectors["sparse"] = models.SparseVector(
                    indices=sp["indices"],
                    values=sp["values"],
                )

            return models.PointStruct(
                id=pt["id"],
                vector=named_vectors,
                payload=pt["payload"],
            )
        except Exception as e:
            logger.error(f"[Sync] PointStruct 변환 실패 (id={pt.get('id')}): {e}")
            return None

    def apply_upsert(self, points_data: List[Dict]):
        """원격에서 받은 포인트를 로컬 Qdrant에 배치 upsert

        ColBERT 멀티벡터 크기 때문에 Qdrant 10MB 제한에 걸릴 수 있으므로
        배치 크기를 동적으로 조절한다.
        """
        UPSERT_BATCH = 5  # 기본 배치 (ColBERT 짧은 문서 기준)
        upserted = 0

        for i in range(0, len(points_data), UPSERT_BATCH):
            batch = points_data[i:i + UPSERT_BATCH]
            structs = [s for s in (self._build_point_struct(pt) for pt in batch) if s]

            if not structs:
                continue

            try:
                self._qdrant.upsert(
                    collection_name=COLLECTION_NAME,
                    points=structs,
                )
                upserted += len(structs)
            except Exception:
                # 배치 실패 시 1개씩 재시도 (10MB 초과 대비)
                for s in structs:
                    try:
                        self._qdrant.upsert(
                            collection_name=COLLECTION_NAME,
                            points=[s],
                        )
                        upserted += 1
                    except Exception as e2:
                        logger.error(f"[Sync] upsert 실패 (id={s.id}): {e2}")

            if upserted % 100 == 0 and upserted > 0:
                logger.info(f"[Sync] upsert 진행: {upserted}/{len(points_data)}")

    def apply_delete(self, local_manifest: Dict, guids: List[str]):
        """로컬에만 있는 포인트 삭제"""
        point_ids = []
        for guid in guids:
            info = local_manifest.get(guid)
            if info and "point_id" in info:
                point_ids.append(info["point_id"])

        if point_ids:
            self._qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=models.PointIdsList(points=point_ids),
            )

    def apply_toggle(self, local_manifest: Dict, remote_manifest: Dict, guids: List[str]):
        """enabled 상태만 변경"""
        for guid in guids:
            local_info = local_manifest.get(guid)
            remote_info = remote_manifest.get(guid)
            if not local_info or not remote_info:
                continue

            self._qdrant.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"enabled": remote_info["enabled"]},
                points=[local_info["point_id"]],
            )

    # ──────────────────────────────────────────────
    # 6. 싱크 실행 (메인 진입점)
    # ──────────────────────────────────────────────
    def run_sync(self) -> Dict[str, int]:
        """싱크 1회 실행

        Returns:
            {"created": N, "updated": N, "deleted": N, "toggled": N}
        """
        if not self.enabled:
            logger.info("[Sync] 싱크 비활성화 상태")
            return {"created": 0, "updated": 0, "deleted": 0, "toggled": 0}

        start = time.time()
        logger.info("[Sync] 싱크 시작...")

        try:
            # 1. manifest 수집
            remote = self.fetch_remote_manifest()
            local = self.collect_local_manifest()
            logger.info(f"[Sync] manifest 수집 완료 (remote={len(remote)}, local={len(local)})")

            # 2. diff 계산
            diff = self.compute_diff(remote, local)
            n_create = len(diff["to_create"])
            n_update = len(diff["to_update"])
            n_delete = len(diff["to_delete"])
            n_toggle = len(diff["to_toggle"])

            if n_create == 0 and n_update == 0 and n_delete == 0 and n_toggle == 0:
                logger.info("[Sync] 변경사항 없음 — 이미 동기화 상태")
                return {"created": 0, "updated": 0, "deleted": 0, "toggled": 0}

            logger.info(
                f"[Sync] diff: 생성={n_create}, 업데이트={n_update}, "
                f"삭제={n_delete}, 토글={n_toggle}"
            )

            # 3. 생성 + 업데이트 대상 포인트 가져오기
            fetch_guids = diff["to_create"] + diff["to_update"]
            if fetch_guids:
                points_data = self.fetch_remote_points(fetch_guids)
                self.apply_upsert(points_data)
                logger.info(f"[Sync] upsert 완료: {len(points_data)}개")

            # 4. 삭제
            if diff["to_delete"]:
                self.apply_delete(local, diff["to_delete"])
                logger.info(f"[Sync] 삭제 완료: {n_delete}개")

            # 5. 토글
            if diff["to_toggle"]:
                self.apply_toggle(local, remote, diff["to_toggle"])
                logger.info(f"[Sync] 토글 완료: {n_toggle}개")

            # 6. RAG 리로드
            if self._llm_handler:
                try:
                    self._llm_handler.reload_embedding(sub_id=SYNC_SUB_ID)
                    logger.info("[Sync] RAG 리로드 완료")
                except Exception as e:
                    logger.warning(f"[Sync] RAG 리로드 실패: {e}")

            elapsed = time.time() - start
            result = {
                "created": n_create,
                "updated": n_update,
                "deleted": n_delete,
                "toggled": n_toggle,
            }
            logger.info(f"[Sync] 싱크 완료 ({elapsed:.1f}초): {result}")
            return result

        except requests.ConnectionError:
            logger.warning(f"[Sync] 메인 서버 연결 실패: {self._main_server_url}")
            return {"created": 0, "updated": 0, "deleted": 0, "toggled": 0}
        except Exception as e:
            logger.error(f"[Sync] 싱크 실패: {e}", exc_info=True)
            return {"created": 0, "updated": 0, "deleted": 0, "toggled": 0}


# ──────────────────────────────────────────────────
# 비동기 래퍼 (부팅 + 주기적 실행)
# ──────────────────────────────────────────────────
_sync_client: Optional[SyncClient] = None


def init_sync(config: ConfigManager, qdrant_client, llm_handler=None) -> Optional[SyncClient]:
    """SyncClient 초기화 (qa_llm.py에서 호출)"""
    global _sync_client
    _sync_client = SyncClient(config, qdrant_client, llm_handler)

    if not _sync_client.enabled:
        logger.info("[Sync] 싱크 비활성화 (config [sync] enabled=False 또는 main_server_url 미설정)")
        return None

    logger.info(
        f"[Sync] 초기화 완료 (서버={_sync_client._main_server_url}, "
        f"주기={_sync_client._interval}초)"
    )
    return _sync_client


async def run_sync_once():
    """싱크 1회 실행 (비동기 래퍼)"""
    if not _sync_client or not _sync_client.enabled:
        return
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_client.run_sync)


async def periodic_sync():
    """주기적 싱크 백그라운드 태스크"""
    if not _sync_client or not _sync_client.enabled:
        return

    interval = _sync_client._interval

    # 부팅 직후 1회 실행
    logger.info("[Sync] 부팅 싱크 시작...")
    await run_sync_once()

    # 이후 주기적 실행
    while True:
        await asyncio.sleep(interval)
        try:
            await run_sync_once()
        except Exception as e:
            logger.error(f"[Sync] 주기적 싱크 실패: {e}", exc_info=True)
