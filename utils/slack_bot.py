"""
Slack Bot Module for ai-agent-question
- ai-agent-slack/slack_integration.py 기반
- 기존 llm_handler를 공유하여 REST API와 동일한 RAG/LLM 사용
- Complete 모드 + Query Validation + 시뮬레이션 스트리밍
"""

import os
import time
import re
import asyncio
import threading
import requests
import tempfile
import base64
import warnings
from typing import Dict, Any, Optional, List

from aibot_validation import Query_validation

# Slack SDK의 text 누락 경고 억제
warnings.filterwarnings('ignore', message='The top-level `text` argument is missing')

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from config_utils import ConfigManager

# === 전역 변수 ===
BOT_USER_ID = None
BOT_ID = None
loop = None
llm_handler = None
response_cache = {}
CACHE_EXPIRY = 3600  # 캐시 유효시간 (초)

# Slack 메시지 최대 길이
MAX_MESSAGE_LENGTH = 3000

# 로딩 애니메이션 스타일
LOADING_STYLES = {
    "dots": ["🤔 답변을 생성하는 중입니다   ", "🤔 답변을 생성하는 중입니다.  ",
             "🤔 답변을 생성하는 중입니다.. ", "🤔 답변을 생성하는 중입니다..."],
    "spinner": ["🌑 ", "🌒 ", "🌓 ", "🌔 ", "🌕 ", "🌖 ", "🌗 ", "🌘 "],
    "working": ["⚙️ 처리 중     ", "⚙️ 처리 중.    ", "⚙️ 처리 중..   ", "⚙️ 처리 중...  "],
    "brain": ["🧠 생각하는 중   ", "🧠 생각하는 중.  ", "🧠 생각하는 중.. ", "🧠 생각하는 중..."]
}


# === 초기화 함수 ===

def init_slack_bot(llm_handler_instance, event_loop=None):
    """
    Slack 봇 초기화

    Args:
        llm_handler_instance: LLM 핸들러 인스턴스
        event_loop: asyncio 이벤트 루프 (없으면 새로 생성)

    Returns:
        slack_app: Slack App 인스턴스
    """
    global llm_handler, loop

    llm_handler = llm_handler_instance

    if event_loop:
        loop = event_loop
    else:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # 설정 로드
    config = ConfigManager()
    slack_config = config.get_slack_config()

    # Slack Bolt 내부 로그 레벨 조정 (세션 갱신 로그 억제)
    import logging
    logging.getLogger("slack_bolt.App").setLevel(logging.WARNING)
    logging.getLogger("slack_bolt.listener").setLevel(logging.WARNING)

    # Slack 앱 초기화
    slack_app = App(token=slack_config['bot_token'])

    return slack_app


# === 유틸리티 함수 ===

def update_loading_message(client, channel_id, message_ts, stop_event, style="dots"):
    """로딩 애니메이션 표시 함수"""
    loading_messages = LOADING_STYLES.get(style, LOADING_STYLES["dots"])
    idx = 0

    while not stop_event.is_set():
        try:
            client.chat_update(
                channel=channel_id,
                ts=message_ts,
                text=loading_messages[idx]
            )
            idx = (idx + 1) % len(loading_messages)
            time.sleep(0.7)
        except SlackApiError as e:
            error_msg = e.response.get('error', 'Unknown error') if hasattr(e, 'response') else str(e)
            print(f"로딩 메시지 Slack API 오류: {error_msg}")
            if error_msg == 'rate_limited':
                time.sleep(1)
        except Exception as e:
            print(f"로딩 메시지 업데이트 오류: {str(e)}")
            break


def split_message(message, max_length=MAX_MESSAGE_LENGTH):
    """긴 메시지를 Slack 제한에 맞게 분할 (코드 블록 경계 보존)"""
    if len(message) <= max_length:
        return [message]

    parts = []
    remaining = message

    while len(remaining) > max_length:
        split_pos = _find_safe_split(remaining, max_length)
        parts.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip('\n')

    if remaining.strip():
        parts.append(remaining)

    return parts


def _find_safe_split(text, max_length):
    """코드 블록을 깨지 않는 안전한 분할 지점 찾기"""
    search_area = text[:max_length]

    # 1) 코드 블록 ``` 이 열려있는지 확인
    code_block_count = search_area.count('```')
    if code_block_count % 2 == 1:
        # 코드 블록이 열려있음 → 마지막 ``` 앞에서 자르기
        last_open = search_area.rfind('```')
        if last_open > max_length // 3:
            # ``` 바로 앞 줄바꿈에서 자르기
            newline_before = search_area.rfind('\n', 0, last_open)
            if newline_before > 0:
                return newline_before + 1
            return last_open

    # 2) 빈 줄(문단 경계)에서 자르기
    last_para = search_area.rfind('\n\n')
    if last_para > max_length // 2:
        return last_para + 1

    # 3) 줄바꿈에서 자르기
    last_newline = search_area.rfind('\n')
    if last_newline > max_length // 2:
        return last_newline + 1

    # 4) 최후의 수단: max_length에서 자르기
    return max_length


def parse_response_mode(text: str) -> tuple:
    """텍스트에서 응답 모드 옵션을 파싱"""
    response_mode = "streaming"

    if "--streaming" in text:
        response_mode = "streaming"
        text = text.replace("--streaming", "").strip()
    elif "--complete" in text:
        response_mode = "complete"
        text = text.replace("--complete", "").strip()
    elif "--stream" in text:
        response_mode = "streaming"
        text = text.replace("--stream", "").strip()

    return response_mode, text


def get_cached_response(query, model_type):
    """캐시 비활성화 — 항상 None 반환"""
    return None


def store_in_cache(query, model_type, response):
    """캐시 비활성화 — 저장하지 않음"""
    pass


def get_user_name(user_id: str, slack_app) -> str:
    """Slack 사용자 ID로 실제 이름 가져오기"""
    try:
        response = slack_app.client.users_info(user=user_id)
        if response["ok"]:
            user = response["user"]
            return user.get("real_name") or user.get("profile", {}).get("display_name") or user.get("name") or user_id
    except Exception as e:
        print(f"사용자 정보 조회 실패: {str(e)}")

    return user_id


def get_thread_history(client, channel_id: str, thread_ts: str, limit: int = 20) -> list:
    """스레드의 대화 히스토리를 가져옴"""
    try:
        result = client.conversations_replies(
            channel=channel_id,
            ts=thread_ts,
            limit=limit
        )

        if result["ok"] and "messages" in result:
            messages = result["messages"]

            history = []
            for msg in messages:
                if msg.get("bot_id"):
                    text = msg.get("text", "")
                    if "*질문[" in text and "]*:" in text:
                        continue
                    history.append({
                        "role": "assistant",
                        "content": text
                    })
                else:
                    text = msg.get("text", "")
                    text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
                    if text:
                        history.append({
                            "role": "user",
                            "content": text
                        })

            if history and history[-1]["role"] == "user":
                history = history[:-1]

            return history

    except Exception as e:
        print(f"스레드 히스토리 조회 실패: {e}")

    return []


# === 이미지 처리 ===

def download_slack_file(url: str, token: str) -> Optional[bytes]:
    """Slack 파일 다운로드"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            print(f"파일 다운로드 실패: {response.status_code}")
            return None
    except Exception as e:
        print(f"파일 다운로드 에러: {e}")
        return None


def process_image_files(files: List[Dict], slack_token: str) -> List[Dict]:
    """이미지 파일 처리"""
    processed_images = []

    for file_info in files:
        if file_info.get("mimetype", "").startswith("image/"):
            url = file_info.get("url_private", file_info.get("url_private_download"))
            if url:
                image_data = download_slack_file(url, slack_token)
                if image_data:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_info.get('filetype', 'jpg')}") as tmp_file:
                        tmp_file.write(image_data)
                        tmp_path = tmp_file.name

                    base64_image = base64.b64encode(image_data).decode('utf-8')

                    processed_images.append({
                        "name": file_info.get("name", "image"),
                        "path": tmp_path,
                        "base64": base64_image,
                        "mimetype": file_info.get("mimetype"),
                        "size": file_info.get("size", 0)
                    })

                    print(f"📷 이미지 처리 완료: {file_info.get('name')} ({file_info.get('size', 0):,} bytes)")

    return processed_images


def create_image_context_message(images: List[Dict], user_text: str) -> str:
    """이미지와 텍스트를 결합한 컨텍스트 메시지 생성"""
    if not images:
        return user_text

    image_info = f"[첨부된 이미지: {len(images)}개]\n"
    for idx, img in enumerate(images, 1):
        image_info += f"- 이미지 {idx}: {img['name']} ({img['size']:,} bytes)\n"

    if user_text:
        return f"{image_info}\n사용자 요청: {user_text}"
    else:
        return f"{image_info}\n이 이미지를 분석해주세요."


def cleanup_temp_images(images: List[Dict]):
    """임시 이미지 파일 삭제"""
    if not images:
        return

    for img in images:
        try:
            if 'path' in img and os.path.exists(img['path']):
                os.remove(img['path'])
                print(f"🗑️ 임시 파일 삭제: {img['path']}")
        except Exception as e:
            print(f"임시 파일 삭제 실패: {e}")


# === 응답 처리 ===

async def _validate_qna_response(full_answer: str, original_question: str, qa_result: dict, user_id: str) -> str:
    """QNA 응답에 대해 쿼리 검증 수행 (api/ai/chats와 동일한 로직)"""
    try:
        from qa_llm import verify_and_fix_query_for_qna
        verification_result = await verify_and_fix_query_for_qna(
            full_answer, original_question, qa_result, user_id
        )
        if verification_result['fixed']:
            print(f"✅ [Slack] QnA 쿼리 수정 완료")
            return verification_result['corrected_response']
        else:
            return verification_result['original_response']
    except ImportError:
        print("⚠️ [Slack] qa_llm 모듈 import 실패, 검증 건너뜀")
        return full_answer
    except Exception as e:
        print(f"⚠️ [Slack] 쿼리 검증 중 오류: {e}")
        return full_answer


def _simulate_streaming_chunks(text: str, chunk_size: int = 80) -> List[str]:
    """완성된 텍스트를 스트리밍처럼 보이기 위한 청크 분할

    문장/줄 경계를 우선으로 분할하여 자연스러운 출력 효과 구현
    """
    if not text:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break

        # 줄바꿈 경계 우선
        newline_pos = remaining.find('\n', 0, chunk_size)
        if newline_pos > 0:
            chunks.append(remaining[:newline_pos + 1])
            remaining = remaining[newline_pos + 1:]
            continue

        # 문장 경계 (. ! ? 뒤 공백)
        best_pos = -1
        for sep in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            pos = remaining.rfind(sep, 0, chunk_size)
            if pos > best_pos:
                best_pos = pos + len(sep)

        if best_pos > chunk_size // 3:
            chunks.append(remaining[:best_pos])
            remaining = remaining[best_pos:]
            continue

        # 공백 경계
        space_pos = remaining.rfind(' ', 0, chunk_size)
        if space_pos > chunk_size // 3:
            chunks.append(remaining[:space_pos + 1])
            remaining = remaining[space_pos + 1:]
            continue

        # 최후: chunk_size에서 자르기
        chunks.append(remaining[:chunk_size])
        remaining = remaining[chunk_size:]

    return chunks


async def process_slack_response(text: str, channel_id: str, thread_ts: str, client, user=None, user_id: str = None, model: str = "gpt", history: list = None, images: list = None):
    """Complete 모드로 응답 생성 → 쿼리 검증 → 시뮬레이션 스트리밍 전송

    1. get_complete_answer()로 전체 응답 생성
    2. verify_and_fix_query_for_qna()로 쿼리 검증/수정
    3. 청크 단위로 Slack 메시지를 업데이트하여 스트리밍 효과 구현
    """
    t_start = time.time()

    # 캐시 확인
    cached_result = get_cached_response(text, model)
    if cached_result:
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            text=cached_result['answer'],
            mrkdwn=True
        )
        print(f"⏱️ [Slack] 캐시 히트, 총 {time.time() - t_start:.2f}s")
        return cached_result

    # 로딩 메시지
    loading_message = client.chat_postMessage(
        channel=channel_id,
        thread_ts=thread_ts,
        text="💭 답변을 생성하는 중입니다..."
    )

    # 로딩 애니메이션
    stop_event = threading.Event()
    loading_thread = threading.Thread(
        target=update_loading_message,
        args=(client, channel_id, loading_message['ts'], stop_event, "brain")
    )
    loading_thread.start()

    try:
        # 모델 설정
        llm_handler.set_model(model)

        # ── 1단계: Complete 모드로 전체 응답 생성 ──
        t_llm_start = time.time()

        request = {
            "question": text,
            "user_id": user_id,
            "channel_id": channel_id,
            "context": 'Slack',
            "sub_id": 1,
            "type": "QNA",
            "response_mode": "complete",
            "locale": "ko",
        }

        if history:
            request["history"] = history

        if images:
            request["images"] = images
            print(f"📷 이미지 {len(images)}개를 요청에 포함")

        qa_result = await llm_handler.get_complete_answer(
            request, user_id, channel_id
        )

        full_response = qa_result.get("answer", "")
        if not full_response:
            full_response = "응답을 생성할 수 없습니다."

        t_llm_end = time.time()
        print(f"⏱️ [Slack] LLM 응답 생성: {t_llm_end - t_llm_start:.2f}s ({len(full_response)}자)")

        # ── 2단계: 쿼리 검증 (api/ai/chats와 동일) ──
        t_validate_start = time.time()

        full_response = await _validate_qna_response(
            full_response, text, qa_result, user_id
        )

        t_validate_end = time.time()
        print(f"⏱️ [Slack] 쿼리 검증: {t_validate_end - t_validate_start:.2f}s")

        # Slack mrkdwn 호환: ```query → ``` 변환 (검증 완료 후)
        full_response = re.sub(r'```query\b', '```', full_response)

        # 로딩 중지
        stop_event.set()
        loading_thread.join()

        # ── 3단계: 시뮬레이션 스트리밍으로 Slack 전송 ──
        t_send_start = time.time()

        SPLIT_THRESHOLD = 2500
        CHUNK_INTERVAL = 0.4  # 청크 간 대기 시간(초)

        message_ts = loading_message['ts']
        current_msg_ts = message_ts
        current_part_text = ""

        chunks = _simulate_streaming_chunks(full_response)

        for chunk in chunks:
            current_part_text += chunk

            # 현재 메시지 파트가 임계값 초과 시 새 메시지로 분할
            if len(current_part_text) > SPLIT_THRESHOLD:
                split_pos = _find_safe_split(current_part_text, SPLIT_THRESHOLD)
                finalized = current_part_text[:split_pos]
                overflow = current_part_text[split_pos:]

                # 현재 메시지 확정
                try:
                    client.chat_update(
                        channel=channel_id, ts=current_msg_ts,
                        text=finalized, mrkdwn=True
                    )
                except SlackApiError:
                    pass

                # 새 메시지 시작
                new_msg = client.chat_postMessage(
                    channel=channel_id, thread_ts=thread_ts,
                    text=overflow if overflow.strip() else "계속...",
                    mrkdwn=True
                )
                current_msg_ts = new_msg['ts']
                current_part_text = overflow
                continue

            # 청크 단위로 메시지 업데이트 (스트리밍 효과)
            try:
                client.chat_update(
                    channel=channel_id, ts=current_msg_ts,
                    text=current_part_text if current_part_text.strip() else "처리 중...",
                    mrkdwn=True
                )
            except SlackApiError as e:
                error_msg = e.response.get('error', '')
                if error_msg == 'ratelimited':
                    await asyncio.sleep(int(e.response.headers.get('Retry-After', 3)))
                elif error_msg == 'msg_too_long':
                    split_pos = _find_safe_split(current_part_text, SPLIT_THRESHOLD)
                    finalized = current_part_text[:split_pos]
                    overflow = current_part_text[split_pos:]
                    try:
                        client.chat_update(
                            channel=channel_id, ts=current_msg_ts,
                            text=finalized, mrkdwn=True
                        )
                    except SlackApiError:
                        pass
                    new_msg = client.chat_postMessage(
                        channel=channel_id, thread_ts=thread_ts,
                        text=overflow if overflow.strip() else "계속...",
                        mrkdwn=True
                    )
                    current_msg_ts = new_msg['ts']
                    current_part_text = overflow
            except Exception:
                pass

            await asyncio.sleep(CHUNK_INTERVAL)

        # 최종 메시지 확정
        final_text = current_part_text if current_part_text and current_part_text.strip() else "응답을 생성할 수 없습니다."
        try:
            client.chat_update(
                channel=channel_id, ts=current_msg_ts,
                text=final_text, mrkdwn=True
            )
        except Exception:
            pass

        t_send_end = time.time()
        t_total = t_send_end - t_start

        print(f"⏱️ [Slack] Slack 전송: {t_send_end - t_send_start:.2f}s ({len(chunks)}청크)")
        print(f"⏱️ [Slack] 전체 소요: {t_total:.2f}s (LLM {t_llm_end - t_llm_start:.2f}s + 검증 {t_validate_end - t_validate_start:.2f}s + 전송 {t_send_end - t_send_start:.2f}s)")

        # 캐시 저장
        cache_data = {
            'answer': full_response,
            'sources': qa_result.get("context_sources", qa_result.get("sources", []))
        }
        store_in_cache(text, model, cache_data)

        if images:
            cleanup_temp_images(images)

        return qa_result

    except Exception as e:
        stop_event.set()
        if 'loading_thread' in locals():
            loading_thread.join()

        if images:
            cleanup_temp_images(images)

        error_message = f"⚠️ 답변 생성 중 오류가 발생했습니다: {str(e)}"
        try:
            client.chat_update(
                channel=channel_id,
                ts=loading_message['ts'],
                text=error_message,
                mrkdwn=True
            )
        except Exception:
            pass
        print(f"Error in process_slack_response: {str(e)}")
        return {"error": str(e)}


async def clean_cache():
    """만료된 캐시 항목 주기적 정리"""
    while True:
        try:
            current_time = time.time()
            expired_keys = []

            for key, item in response_cache.items():
                if current_time - item['timestamp'] > CACHE_EXPIRY:
                    expired_keys.append(key)

            for key in expired_keys:
                del response_cache[key]

            if expired_keys:
                print(f"[Slack] 캐시에서 {len(expired_keys)}개 항목 제거됨")

        except Exception as e:
            print(f"[Slack] 캐시 정리 중 오류: {str(e)}")

        await asyncio.sleep(3600)


# === Slack 핸들러 등록 ===

def register_slack_handlers(slack_app):
    """Slack 앱에 명령어 및 이벤트 핸들러 등록"""
    config = ConfigManager()

    # GPT 명령어 핸들러
    @slack_app.command("/gpt")
    def handle_gpt_command(ack, body, say, client):
        ack()

        text = body.get("text", "").strip()
        channel_id = body["channel_id"]
        user_id = body["user_id"]
        user = get_user_name(user_id, slack_app)

        if not text:
            say("사용법: `/gpt [질문]`\n예시: `/gpt 안녕하세요`")
            return

        # 레거시 플래그 제거 (호환성)
        _, clean_text = parse_response_mode(text)

        result = client.chat_postMessage(
            channel=channel_id,
            text=f"🤖 *질문[GPT]*: {clean_text}"
        )

        if result['ok']:
            thread_ts = result['ts']
            asyncio.run_coroutine_threadsafe(
                process_slack_response(clean_text, channel_id, thread_ts, client, user, user_id, "gpt"),
                loop
            )

    # GPT-OSS 명령어 핸들러
    @slack_app.command("/gpt-oss")
    def handle_gpt_oss_command(ack, body, say, client):
        ack()

        text = body.get("text", "").strip()
        channel_id = body["channel_id"]
        user_id = body["user_id"]
        user = get_user_name(user_id, slack_app)

        if not text:
            say("질문을 입력해주세요!")
            return

        _, clean_text = parse_response_mode(text)

        result = client.chat_postMessage(
            channel=channel_id,
            text=f"🔥 *질문[GPT-OSS]*: {clean_text}"
        )

        if result['ok']:
            thread_ts = result['ts']
            asyncio.run_coroutine_threadsafe(
                process_slack_response(clean_text, channel_id, thread_ts, client, user, user_id, "gpt-oss"),
                loop
            )

    # Llama 명령어 핸들러
    @slack_app.command("/llama")
    def handle_llama_command(ack, body, say, client):
        ack()

        text = body.get("text", "").strip()
        channel_id = body["channel_id"]
        user_id = body["user_id"]
        user = get_user_name(user_id, slack_app)

        if not text:
            say("질문을 입력해주세요!")
            return

        _, clean_text = parse_response_mode(text)

        result = client.chat_postMessage(
            channel=channel_id,
            text=f"🦙 *질문[LLAMA]*: {clean_text}"
        )

        if result['ok']:
            thread_ts = result['ts']
            asyncio.run_coroutine_threadsafe(
                process_slack_response(clean_text, channel_id, thread_ts, client, user, user_id, "llama"),
                loop
            )

    # 봇 멘션 처리
    @slack_app.event("app_mention")
    def handle_app_mention(body, say, client):
        event = body["event"]
        channel_id = event["channel"]
        user_id = event["user"]
        user = get_user_name(user_id, slack_app)
        text = event["text"]

        # 봇 ID 패턴 제거
        text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()

        # 이미지 파일 확인 및 처리
        files = event.get("files", [])
        images = None
        if files:
            slack_token = config.get_slack_config().get("bot_token")
            images = process_image_files(files, slack_token)
            if images:
                print(f"📷 {len(images)}개의 이미지 감지됨 (멘션)")
                if not text:
                    text = "이 이미지를 분석해주세요."

        if not text and not images:
            client.chat_postMessage(
                channel=channel_id,
                text="질문을 입력해주세요!",
                thread_ts=event.get("thread_ts")
            )
            return

        # 모델 선택 (기본: GPT)
        model = "gpt"
        if "oss" in text.lower()[:20]:
            model = "gpt-oss"
        elif "llama" in text.lower()[:20]:
            model = "llama"

        # 스레드 컨텍스트 확인
        existing_thread_ts = event.get("thread_ts")
        thread_ts = existing_thread_ts if existing_thread_ts else event.get("ts")

        # 스레드 내 멘션인 경우 히스토리 가져오기
        history = None
        if existing_thread_ts:
            history = get_thread_history(client, channel_id, existing_thread_ts)
            if history:
                print(f"📚 스레드 히스토리 수집 (멘션): {len(history)}개 메시지")

        asyncio.run_coroutine_threadsafe(
            process_slack_response(text, channel_id, thread_ts, client, user, user_id, model, history=history, images=images),
            loop
        )

    # 메시지 이벤트 처리 (DM 및 스레드)
    @slack_app.event("message")
    def handle_message_events(body, logger, client):
        event = body.get("event", {})

        # 봇 메시지는 무시
        if event.get("bot_id"):
            return

        channel_id = event.get("channel")
        user_id = event.get("user")
        text = event.get("text", "").strip()

        # 이미지 파일 확인 및 처리
        files = event.get("files", [])
        images = None
        if files:
            slack_token = config.get_slack_config().get("bot_token")
            images = process_image_files(files, slack_token)
            if images:
                print(f"📷 {len(images)}개의 이미지 감지됨")
                if not text:
                    text = "이 이미지를 분석해주세요."

        # 스레드 내 메시지 처리
        thread_ts = event.get("thread_ts")
        if thread_ts:
            is_dm = event.get("channel_type") == "im"
            has_mention = re.search(r'<@[A-Z0-9]+>', text)

            if has_mention:
                return  # 멘션은 app_mention 이벤트에서 처리

            # DM 스레드에서 멘션이 없는 일반 메시지만 처리
            if is_dm:
                cleaned_text = text

                if not cleaned_text:
                    client.chat_postMessage(
                        channel=channel_id,
                        text="질문을 입력해주세요!",
                        thread_ts=thread_ts
                    )
                    return

                user = get_user_name(user_id, slack_app)

                history = get_thread_history(client, channel_id, thread_ts)
                if history:
                    print(f"📚 스레드 히스토리 수집: {len(history)}개 메시지")

                asyncio.run_coroutine_threadsafe(
                    process_slack_response(cleaned_text, channel_id, thread_ts, client, user, user_id, "gpt", history=history, images=images),
                    loop
                )
            return

        # DM 처리
        if event.get("channel_type") == "im":
            if not text and not images:
                return

            has_mention = re.search(r'<@[A-Z0-9]+>', text)
            if has_mention:
                return  # 멘션은 app_mention 이벤트에서 처리

            user = get_user_name(user_id, slack_app)
            thread_ts = event.get("ts")

            asyncio.run_coroutine_threadsafe(
                process_slack_response(text, channel_id, thread_ts, client, user, user_id, "gpt", images=images),
                loop
            )

    # 캐시 초기화 명령어
    @slack_app.command("/clear")
    def handle_clear_command(ack, body, say, client):
        ack()

        global response_cache
        cache_size = len(response_cache)
        response_cache.clear()

        client.chat_postMessage(
            channel=body["channel_id"],
            text=f"✅ 캐시가 초기화되었습니다. (제거된 항목: {cache_size}개)"
        )

    print("✅ Slack 핸들러 등록 완료")


# === Slack 봇 실행 ===

def run_slack_bot(slack_app, app_token):
    """Slack 봇 실행 (SocketModeHandler, blocking)"""
    global BOT_USER_ID, BOT_ID

    print("⚡️ Slack 봇 시작...")

    # 봇 ID 가져오기
    try:
        auth_response = slack_app.client.auth_test()
        BOT_USER_ID = auth_response.get("user_id")
        BOT_ID = auth_response.get("bot_id")
        print(f"✅ 봇 ID 확인: User={BOT_USER_ID}, Bot={BOT_ID}")
    except Exception as e:
        print(f"⚠️ 봇 ID 확인 실패: {str(e)}")

    # 캐시 정리 태스크 시작
    if loop:
        asyncio.run_coroutine_threadsafe(clean_cache(), loop)

    # Socket Mode 핸들러로 실행
    handler = SocketModeHandler(slack_app, app_token)

    print("✅ Slack 봇이 성공적으로 시작되었습니다.")
    print("📌 사용 가능한 명령어:")
    print("  - /gpt [질문] - GPT 모델로 질문")
    print("  - /gpt-oss [질문] - GPT-OSS 모델로 질문")
    print("  - /llama [질문] - Llama 모델로 질문")
    print("  - /clear - 캐시 초기화")
    print("  - @봇멘션 [질문] - 봇에게 직접 질문")
    print("  - DM으로 직접 대화 가능")

    handler.start()
