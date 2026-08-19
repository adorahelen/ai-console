
import os
import json
import time
import uuid
import re
import asyncio
from typing import List, Dict, Optional, Any, AsyncGenerator, Tuple, Union
from dataclasses import dataclass
import tiktoken
import datetime
import boto3
from openai import OpenAI
from config_utils import ConfigManager
from aibot_rag_module_BGE import RAGSystemBGE
from aibot_logger import ChatLogger, LOGGERMODE
import textwrap
import logging

from handler_registry import (
    HANDLER_CLASSES,
    HANDLER_PRIORITY,
    FALLBACK_MODEL,
    LEGACY_ATTR_TO_KEY,
)


from aibot_validation import Query_validation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aibot_llm")

class SecurityAnalysisResult:
    def __init__(self, classification: str, analysis: str, updated_queue: List[str]):
        self.classification = classification
        self.analysis = analysis
        self.updated_queue = updated_queue

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: float
    message_id: str

class ChatSession:
    def __init__(self, session_id: str = None, max_messages: int = 10):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
        self.last_activity = time.time()

    def add_message(self, role: str, content: str) -> ChatMessage:
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            message_id=str(uuid.uuid4())
        )

        self.messages.append(message)
        self.last_activity = time.time()

        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

        return message

    def get_chat_history(self, max_tokens: int = 2000) -> List[Dict[str, str]]:
        history = []
        token_count = 0
        from aibot_embedding import get_encoding
        tokenizer = get_encoding("cl100k_base")

        for message in reversed(self.messages):
            tokens = len(tokenizer.encode(message.content))

            if token_count + tokens > max_tokens:
                break

            history.insert(0, {
                "role": message.role,
                "content": message.content
            })

            token_count += tokens

        return history

    def format_chat_history_for_llama(self, max_tokens: int = 2000) -> str:
        messages = self.get_chat_history(max_tokens)
        formatted_history = ""

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                formatted_history += f"<human>: {content}\n\n"
            elif role == "assistant":
                formatted_history += f"<assistant>: {content}\n\n"

        return formatted_history

    def clear(self) -> None:
        self.messages = []


def _handler_property(key: str) -> property:
    """self.handlers[key] 를 self.<legacy_name>_handler 로 노출 (backward-compat).

    qa_llm.py 등 외부 코드가 llm_handler.openai_handler 식으로 직접 접근하는
    부분을 그대로 유지하기 위함. 새 호출자는 self.handlers[key] 권장.
    """
    def getter(self): return self.handlers.get(key)
    def setter(self, value):
        if value is None:
            self.handlers.pop(key, None)
        else:
            self.handlers[key] = value
    return property(getter, setter)


class LLMHandler:
    openai_handler = _handler_property("gpt")
    gptoss_handler = _handler_property("gpt-oss")
    qwen_handler   = _handler_property("qwen")
    gemma_handler  = _handler_property("gemma")
    llama_handler  = _handler_property("llama")

    def __init__(self, prompt_type: str = "llama", use_rag: bool = True, file_extension: str = "yaml", use_kg: bool = True,
        use_db: bool = False):
        self.config_manager = ConfigManager()
        self.prompt_type = prompt_type
        self.use_rag = use_rag
        self.use_kg = use_kg
        self.file_extension = file_extension

        self.use_db = use_db

        if use_db is None:
            db_config = self.config_manager.get_db_config()
            self.use_db = db_config.get('use_db_mode', False)
        else:
            self.use_db = use_db


        self.query_validator = Query_validation()

        self.chat_sessions = {}

        self.intent_analyzer = None

        self.rag_system = None

        # default_model을 먼저 정의
        default_model = self.config_manager.get_model_config().get('model', 'llama')

        if self.use_rag:
            # BGE-M3 단일 백엔드 (legacy RAGSystem 제거) — 설정이 꺼져 있으면 RAG 없이
            # 조용히 뜨지 않도록 try 진입 전에 즉시 실패한다. getboolean = true/True/1/yes 모두 허용.
            if not self.config_manager.config.getboolean('embedding', 'use_bge_mode', fallback=False):
                raise RuntimeError(
                    "legacy RAG(비-BGE)은 제거되었습니다 — config [embedding] use_bge_mode = True 필요"
                )
            try:
                if default_model == "gpt":
                    embedding_folder = "qa_embeddings_gpt"
                elif default_model in ("gpt-oss", "qwen", "gemma"):
                    embedding_folder = "qa_embeddings"
                else:
                    embedding_folder = "qa_embeddings"

                logger.info(f"RAG initialization for {default_model.upper()} model: embedding_folder='{embedding_folder}'")

                db_manager = None
                if self.use_db:
                    from aibot_db_manager import AibotDBManager
                    from aibot_db_command import SQL_QUERIES

                    db_manager = AibotDBManager(ConfigManager(), SQL_QUERIES)
                    logger.info(f"Initializing RAG system in DB mode")
                else:
                    logger.info(f"Initializing RAG system in file mode")

                logger.info(f"🚀 Using BGE-M3 RAG system for {default_model.upper()} model")
                logger.info("   📋 BGE 모드에서는 지식그래프가 비활성화됩니다")
                # BGE 모드에서는 지식그래프 사용 안함
                self.use_kg = False

                self.rag_system = RAGSystemBGE(
                    file_extension=file_extension,
                    embedding_folder=embedding_folder,
                    use_db=self.use_db,
                    sub_id=None,
                    db_manager=db_manager,
                    config_manager=self.config_manager
                )

                logger.info(f"RAG system initialization complete for {default_model.upper()}")

                if hasattr(self.rag_system, 'set_llm_handler'):
                    self.rag_system.set_llm_handler(self)

                if not self.use_db and hasattr(self.rag_system, 'initialize_ann_index'):
                    self.rag_system.initialize_ann_index()

            except Exception as e:
                logger.error(f"RAG system initialization failed: {str(e)}")
                self.rag_system = None
                self.use_rag = False
                self.use_kg = False

        if self.use_kg:
            kg_available = self.rag_system and hasattr(self.rag_system, 'kg_available') and self.rag_system.kg_available

            if not kg_available:
                logger.warning("Knowledge graph is not available.")
                self.use_kg = False
            else:
                logger.info(f"Knowledge graph activated")

        self.handlers: Dict[str, Any] = {}

        primary_key = default_model if default_model in HANDLER_CLASSES else FALLBACK_MODEL
        primary = HANDLER_CLASSES[primary_key](self.rag_system, self.use_kg, None)
        self.handlers[primary_key] = primary
        self.model = primary_key if primary.available else None

        if "gpt" not in self.handlers:
            try:
                openai_api_key = self.config_manager.get_openai_config().get('api_key', '') or ''
                key_looks_valid = openai_api_key and not openai_api_key.startswith('YOUR-')
                if key_looks_valid:
                    candidate = HANDLER_CLASSES["gpt"](self.rag_system, self.use_kg, None)
                    if candidate.available and candidate.validate_api_key():
                        self.handlers["gpt"] = candidate
                        logger.info(f"OpenAI handler loaded as secondary (default_model={default_model}, multi-model routing enabled)")
                    else:
                        logger.warning("OpenAI API key present but validation failed — secondary handler not loaded")
            except Exception as e:
                logger.warning(f"OpenAI secondary handler load skipped: {e}")

        try:
            from aibot_intent_analyzer import IntentAnalyzer

            active_handler = None
            for key in HANDLER_PRIORITY:
                h = self.handlers.get(key)
                if h and getattr(h, 'available', False):
                    active_handler = h
                    break

            if active_handler:
                self.intent_analyzer = IntentAnalyzer(llm_client=active_handler, config_manager=self.config_manager)
                logger.info(f"Intent analyzer initialized with LLM: {active_handler.__class__.__name__}")
            else:
                self.intent_analyzer = IntentAnalyzer(llm_client=None, config_manager=self.config_manager)
                logger.warning(f"No active handler - intent analyzer initialized in heuristic mode")

        except Exception as e:
            logger.warning(f"Intent analyzer initialization failed: {str(e)}")
            self.intent_analyzer = None

        if self.intent_analyzer:
            try:
                for h in self.handlers.values():
                    h.intent_analyzer = self.intent_analyzer

                # llm_client/connect_llm 호출 방식이 핸들러별로 다름 — 원본 순서 보존
                # (마지막 write 가 llm_client 를 결정하므로 순서 의미 있음)
                if (h := self.handlers.get("gpt")):
                    if hasattr(h, 'client') and h.client:
                        self.intent_analyzer.llm_client = h.client
                    logger.info("OpenAI handler intent analyzer setup complete")
                if (h := self.handlers.get("llama")):
                    self.intent_analyzer.llm_client = h
                    logger.info("Llama handler intent analyzer setup complete")
                for key in ("gpt-oss", "qwen", "gemma"):
                    if (h := self.handlers.get(key)):
                        success = self.intent_analyzer.connect_llm(h)
                        logger.info(f"{h.__class__.__name__} intent analyzer setup: {success}")
            except Exception as e:
                logger.warning(f"Handler intent analyzer setup failed: {e}")

        if self.rag_system and self.intent_analyzer:
            try:
                if hasattr(self.rag_system, 'intent_analyzer'):
                    self.rag_system.intent_analyzer = self.intent_analyzer
                    logger.info("RAG system intent analyzer connected")

                if hasattr(self.rag_system, 'set_intent_analyzer'):
                    self.rag_system.set_intent_analyzer(self.intent_analyzer)
                    logger.info("RAG system set_intent_analyzer call complete")

            except Exception as e:
                logger.warning(f"RAG intent analyzer connection failed: {e}")


        logger.info(f"🚀 LLM 핸들러 초기화 완료 (DEFAULT_MODEL 기반)")
        logger.info(f"   - 설정된 DEFAULT_MODEL: {default_model}")
        logger.info(f"   - 전달받은 prompt_type: {prompt_type}")
        logger.info(f"   - 활성 모델: {self.model}")
        logger.info(f"   - 사용된 임베딩 폴더: {embedding_folder if 'embedding_folder' in locals() else 'N/A'}")
        for key in HANDLER_PRIORITY:
            logger.info(f"   - {key} 핸들러: {'✅ 로드됨' if key in self.handlers else '❌ 미로드'}")
        logger.info(f"   - RAG 시스템: {'✅ 활성화' if self.rag_system else '❌ 비활성화'}")
        logger.info(f"   - 지식 그래프: {'✅ 활성화' if self.use_kg else '❌ 비활성화'}")
        logger.info(f"   - 의도 분석기: {'✅ 활성화' if self.intent_analyzer else '❌ 비활성화'}")


    def set_model(self, model_name: str) -> bool:
        h = self.handlers.get(model_name)
        if h and h.available:
            self.model = model_name
            return True
        return False

    def clear_chat_history(self) -> str:
        for h in self.handlers.values():
            h.clear_chat_history()
        self.chat_sessions.clear()
        return "모든 대화 기록이 초기화되었습니다."

    def get_last_sources(self) -> List[str]:
        h = self.handlers.get(self.model)
        return h.get_last_sources() if h else []

    def _get_response_mode_by_intent(self, question_type: str = None) -> str:
        if not question_type:
            return self.config_manager.get_option_config().get('response', 'streaming')

        intent_upper = question_type.upper()

        if intent_upper == 'QNA':
            return 'streaming'
        elif intent_upper in ['ACTION', 'PLAN','PLAYBOOK']:
            return 'complete'
        else:
            return self.config_manager.get_option_config().get('response', 'streaming')


    def _get_active_handler(self):
        h = self.handlers.get(self.model)
        return h if (h and h.available) else None

    def get_model_status(self) -> Dict[str, Any]:
        status = {
            "current_model": self.model,
            "local_model_enabled": os.getenv("model_opt", "on").lower() == "on",
            "rag_enabled": self.use_rag,
            "kg_enabled": self.use_kg,
            "intent_analyzer_enabled": bool(self.intent_analyzer),
            "active_sessions": len(self.chat_sessions)
        }
        # 호환성: openai_available / gptoss_available / qwen_available / gemma_available / llama_available
        for legacy_attr, key in LEGACY_ATTR_TO_KEY.items():
            h = self.handlers.get(key)
            status[legacy_attr.replace("_handler", "_available")] = h.available if h else False
        return status

    def get_system_info(self) -> str:
        status = self.get_model_status()
        intent_status = self.get_intent_analysis_status()

        info = f"""
            🚀 **LLM 핸들러 시스템 정보**

            **모델 상태:**
            - 현재 모델: {status['current_model']}
            - OpenAI 사용 가능: {'✅' if status['openai_available'] else '❌'}
            - Llama 사용 가능: {'✅' if status['llama_available'] else '❌'}
            - GPT-OSS 사용 가능: {'✅' if status['gptoss_available'] else '❌'}
            - Qwen 사용 가능: {'✅' if status['qwen_available'] else '❌'}
            - Gemma 사용 가능: {'✅' if status['gemma_available'] else '❌'}
            - 로컬 모델 활성화: {'✅' if status['local_model_enabled'] else '❌'}

            **기능 상태:**
            - RAG 시스템: {'✅' if status['rag_enabled'] else '❌'}
            - 지식 그래프: {'✅' if status['kg_enabled'] else '❌'}
            - 의도 분석기: {'✅' if intent_status['intent_analyzer_enabled'] else '❌'}
            - 의도 분석기 사용 가능: {'✅' if intent_status['intent_analyzer_available'] else '❌'}

            **세션 정보:**
            - 활성 채팅 세션: {status['active_sessions']}개

            **프롬프트 유형:** {self.prompt_type}

            **🎯 GPT-OSS 추가:**
            - 🤖 20B 파라미터 오픈소스 모델
            - 🔧 GGUF 형식 지원
            - 🎯 GPT 스타일 프롬프트 사용
            - 🚀 로컬 실행으로 빠른 응답
        """.strip()

        return info

    async def get_streaming_answer(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None) -> Dict:
        handler = self._get_active_handler()
        if not handler:
            async def error_stream():
                yield "사용 가능한 AI 모델이 없습니다."

            return {
                "answer_stream": error_stream(),
                "context_sources": []
            }

        try:
            return await handler.process_question(request, user_id, channel_id, connection, thread_ts)

        except Exception as e:
            logger.error(f"스트리밍 응답 처리 중 오류: {e}")

            async def error_stream():
                yield f"스트리밍 응답 생성 중 오류: {str(e)}"

            return {
                "answer_stream": error_stream(),
                "context_sources": []
            }

    async def get_complete_answer(self, request, user_id=None, channel_id=None, connection=None, thread_ts=None) -> Dict:
        if hasattr(self, 'log_hardware_request'):
            self.log_hardware_request()

        try:
            handler = self._get_active_handler()
            if not handler:
                if isinstance(request, dict):
                    question_type = request.get("type", None)
                else:
                    question_type = None

                return {
                    "answer": "사용 가능한 AI 모델이 없습니다.",
                    "sources": [],
                    "intent": "error",
                    "question_type": question_type
                }

            if isinstance(request, dict):
                question = request.get("question", "")
                question_type = request.get("type", None)

                if question_type in ["PLAN", "auto"]:
                    print(f"🔒 [워크플로우 모드] PLAN/auto 요청 감지, 워크플로우로 처리: {question}")

                    if question_type == "auto":
                        print(f"🔒 [AUTO→PLAN 변환] auto 모드를 PLAN 워크플로우로 처리")

                    if hasattr(self, 'rag_system') and self.rag_system:
                        try:
                            rag_result = await self.rag_system.get_related_documents({
                                "question": question,
                                "type": "PLAN"
                            })

                            context, source_files = "", []
                            related_concepts, concepts = [], []
                            intent_info = {"primary_intent": "PLAN", "distribution": None, "is_multi": False}

                            if len(rag_result) >= 2:
                                context, source_files = rag_result[0], rag_result[1]
                                if len(rag_result) > 2:
                                    related_concepts = rag_result[2] if rag_result[2] else []
                                if len(rag_result) > 3:
                                    concepts = rag_result[3] if rag_result[3] else []
                                if len(rag_result) > 4:
                                    intent_info = rag_result[4] if rag_result[4] else intent_info

                            return await handler.generate_complete(
                                {
                                    "question": question,
                                    "type": "PLAN",
                                    "history": request.get("history", [])
                                },
                                user_id, channel_id, connection, thread_ts,
                                context, source_files, related_concepts, concepts, intent_info
                            )
                        except Exception as e:
                            print(f"🔒 [워크플로우 RAG 오류] {e}")
                            return await handler.generate_complete(
                                {
                                    "question": question,
                                    "type": "PLAN",
                                    "history": request.get("history", [])
                                },
                                user_id, channel_id, connection, thread_ts
                            )
                    else:
                        return await handler.generate_complete(
                            {
                                "question": question,
                                "type": "PLAN",
                                "history": request.get("history", [])
                            },
                            user_id, channel_id, connection, thread_ts
                        )

                if False:
                    if hasattr(self, 'rag_system') and self.rag_system:
                        try:
                            rag_result = await self.rag_system.get_related_documents({
                                "question": question,
                                "type": "ACTION"
                            })

                            if len(rag_result) >= 2:
                                context, source_files = rag_result[0], rag_result[1]

                                if False:
                                    print(f"🔒 [보안 워크플로우 감지] 자동 조사 엔진으로 라우팅")

                                    if question_type == "auto":
                                        print(f"🔒 [AUTO→PLAN 변환] 다단계 질문을 PLAN 모드로 강제 처리")


                                    print(f"🔒 [PLAN 단계] 참조문서와 함께 단계 분해 실행")
                                    plan_result = await handler.generate_complete(
                                        {
                                            "question": question,
                                            "type": "PLAN",
                                            "history": request.get("history", [])
                                        },
                                        user_id, channel_id, connection, thread_ts,
                                        context, source_files, rag_result[2] if len(rag_result) > 2 else [],
                                        rag_result[3] if len(rag_result) > 3 else [],
                                        rag_result[4] if len(rag_result) > 4 else None
                                    )

                                    if not plan_result or not plan_result.get("answer"):
                                        print(f"🔒 [워크플로우 오류] PLAN 단계 실패")
                                        return await handler.generate_complete(request, user_id, channel_id, connection, thread_ts)

                                    plan_steps = plan_result.get("answer", "")
                                    print(f"🔒 [PLAN 완료] 분해된 단계들: {plan_steps[:200]}...")

                                    print(f"🔒 [ACTION 단계] 쿼리 생성 (워크플로우 전용)")
                                    action_result = await handler.generate_complete(
                                        {
                                            "question": f"다음 계획에 따라 첫 번째 단계의 쿼리를 생성하세요:\n\n계획: {plan_steps}\n\n원본 질문: {question}",
                                            "type": "WORKFLOW_ACTION",
                                            "history": request.get("history", [])
                                        },
                                        user_id, channel_id, connection, thread_ts,
                                        context, source_files, rag_result[2] if len(rag_result) > 2 else [],
                                        rag_result[3] if len(rag_result) > 3 else [],
                                        rag_result[4] if len(rag_result) > 4 else None
                                    )

                                    if not action_result or not action_result.get("answer"):
                                        print(f"🔒 [워크플로우 오류] ACTION 단계 실패")
                                        return plan_result

                                    generated_query = action_result.get("answer", "")
                                    print(f"🔒 [쿼리 생성 완료] {generated_query[:100]}...")

                                    simulated_result = f"[시뮬레이션 결과] {question}에 대한 데이터: 검색된 로그 100건, 의심스러운 패턴 3개 발견"
                                    print(f"🔒 [쿼리 실행 완료] 시뮬레이션 결과 생성")

                                    print(f"🔒 [POST-ACTION 단계] 결과 분석 (워크플로우 전용)")
                                    initial_result = await handler.generate_complete(
                                        {
                                            "question": f"다음 쿼리 결과를 분석해주세요:\n\n원본 질문: {question}\n계획: {plan_steps}\n실행된 쿼리: {generated_query}\n쿼리 결과: {simulated_result}",
                                            "type": "WORKFLOW_POST_ACTION",
                                            "history": request.get("history", [])
                                        },
                                        user_id, channel_id, connection, thread_ts,
                                        context, source_files, rag_result[2] if len(rag_result) > 2 else [],
                                        rag_result[3] if len(rag_result) > 3 else [],
                                        rag_result[4] if len(rag_result) > 4 else None
                                    )

                                    if initial_result and initial_result.get("answer"):
                                        analysis_text = initial_result.get("answer", "")
                                        security_analysis = self._parse_security_analysis(analysis_text)

                                        if security_analysis and security_analysis.updated_queue:
                                            print(f"🔒 [자동 조사 시작] {len(security_analysis.updated_queue)}개 후속 조사 자동 실행")

                                            print(f"🔒 [워크플로우 계속] {len(security_analysis.updated_queue)}개 후속 조사 단계 실행 시작")

                                            if security_analysis.updated_queue:
                                                next_step = security_analysis.updated_queue[0]
                                                print(f"🚀 [다음 단계 실행] {next_step}")

                                                try:
                                                    next_step_result = await handler.generate_complete(
                                                        {
                                                            "question": next_step,
                                                            "type": "ACTION",
                                                            "history": request.get("history", [])
                                                        },
                                                        user_id, channel_id, connection, thread_ts,
                                                        context, source_files, rag_result[2] if len(rag_result) > 2 else [],
                                                        rag_result[3] if len(rag_result) > 3 else [],
                                                        {"intent": "action"}
                                                    )

                                                    if next_step_result and next_step_result.get("answer"):
                                                        print(f"✅ [다음 단계 완료] {next_step[:50]}...")
                                                        final_analysis = f"{security_analysis.analysis}\n\n📋 **다음 단계 실행됨**: {next_step}\n\n{next_step_result.get('answer', '')}"
                                                    else:
                                                        print(f"❌ [다음 단계 실패] {next_step[:50]}...")
                                                        final_analysis = security_analysis.analysis

                                                except Exception as e:
                                                    print(f"❌ [다음 단계 오류] {e}")
                                                    final_analysis = security_analysis.analysis
                                            else:
                                                final_analysis = security_analysis.analysis

                                            final_result = {
                                                "success": True,
                                                "final_analysis": final_analysis,
                                                "classification": security_analysis.classification,
                                                "updated_queue": security_analysis.updated_queue[1:] if len(security_analysis.updated_queue) > 1 else [],
                                                "total_steps_executed": 2 if security_analysis.updated_queue else 1
                                            }

                                            if final_result.get("success"):
                                                return {
                                                    "answer": final_result.get("final_analysis", ""),
                                                    "sources": [],
                                                    "intent": "POST-ACTION",
                                                    "question_type": "POST-ACTION",
                                                    "security_workflow": True,
                                                    "investigation_summary": {
                                                        "total_steps": final_result.get("total_steps_executed", 0),
                                                        "completed_steps": final_result.get("completed_steps", []),
                                                        "investigation_depth": final_result.get("investigation_depth", 0)
                                                    }
                                                }
                                            else:
                                                return {
                                                    "answer": f"{analysis_text}\n\n**자동 조사 실패**: {final_result.get('error', '알 수 없는 오류')}",
                                                    "sources": initial_result.get("sources", []),
                                                    "intent": "POST-ACTION",
                                                    "question_type": "POST-ACTION",
                                                    "security_workflow": True,
                                                    "workflow_status": "investigation_failed"
                                                }

                                        print(f"🔒 [조사 완료] 후속 조사 없음, POST-ACTION 결과로 종료")
                                        return {
                                            "answer": analysis_text,
                                            "sources": initial_result.get("sources", []),
                                            "intent": "POST-ACTION",
                                            "question_type": "POST-ACTION",
                                            "security_workflow": True,
                                            "workflow_status": "completed"
                                        }

                                    else:
                                        print(f"🔒 [POST-ACTION 실패] ACTION 결과로 종료")
                                        return {
                                            "answer": f"## 보안 조사 결과\n\n**계획:**\n{plan_steps}\n\n**생성된 쿼리:**\n```\n{generated_query}\n```\n\n**쿼리 결과:**\n{simulated_result}\n\n*POST-ACTION 분석이 실패하여 쿼리 실행 결과까지만 제공됩니다.*",
                                            "sources": action_result.get("sources", []),
                                            "intent": "ACTION",
                                            "question_type": "ACTION",
                                            "security_workflow": True,
                                            "workflow_status": "partial_completion"
                                        }

                        except Exception as e:
                            print(f"🔒 [보안 워크플로우 오류] RAG 검색 실패: {e}")

                    print(f"🔒 [일반 PLAN] 워크플로우 비활성화됨, 일반 처리 진행")

            if not isinstance(request, dict):
                original_request = str(request)
                request = {"question": original_request}

            question_type = request.get("type", "")
            if question_type and question_type.upper() in ['ACTION', 'PLAN']:
                app_context = request.get("apps", None)
                if app_context:
                    request["app_context"] = app_context
                    print(f"📱 [앱 컨텍스트] {question_type.upper()} 의도 감지 - 설치된 앱 정보를 핸들러에 전달합니다")

            return await handler.generate_complete(request, user_id, channel_id, connection, thread_ts)

        except Exception as e:
            print(f"LLMHandler get_complete_answer 오류: {str(e)}")

            if isinstance(request, dict):
                question_type = request.get("type", None)
            else:
                question_type = None

            return {
                "answer": f"응답 생성 중 오류가 발생했습니다: {str(e)}",
                "sources": [],
                "intent": "error",
                "question_type": question_type
            }

    async def process_question(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None) -> Dict:
        if hasattr(self, 'log_hardware_request'):
            self.log_hardware_request()

        if isinstance(request, dict):
            query = request.get("question", "")
            question_type = request.get("type", None)
            sub_id = request.get("sub_id", None)  
        else:
            query = str(request)
            question_type = None
            sub_id = None

        if sub_id is None:
            raise ValueError("request에 sub_id가 필요합니다")

        print(f"📨 [LLM Handler] 구독 ID: {sub_id}, 질문 타입: {question_type}, 질문: {query[:50]}...")

        # 요청 단위 오버라이드 — /api/query/stream(stream_tokens)처럼 호출자가 모드를
        # 명시해야 하는 경로용. 없으면 종전대로 intent/config 규칙.
        override_mode = request.get("response_mode") if isinstance(request, dict) else None
        response_mode = override_mode or self._get_response_mode_by_intent(question_type)

        if question_type:
            if question_type.upper() == 'QNA':
                logger.info(f"🎯 QNA 의도 감지: 스트리밍 모드 강제 적용")
            elif question_type.upper() in ['ACTION', 'PLAN', 'PLAYBOOK']:
                logger.info(f"🎯 {question_type.upper()} 의도 감지: 완전한 응답 모드 강제 적용")
            else:
                logger.info(f"🎯 기타 의도({question_type}): 설정값({response_mode}) 사용")

        handler = self._get_active_handler()
        if not handler:
            async def error_stream():
                yield "사용 가능한 AI 모델이 없습니다."

            return {
                "answer_stream": error_stream(),
                "context_sources": []
            }

        if not isinstance(request, dict):
            original_request = str(request)
            request = {"question": original_request}

        if question_type and question_type.upper() in ['ACTION', 'PLAN']:
            app_context = request.get("apps", None)
            if app_context:
                original_question = request.get("question", "")
                request["question"] = original_question + app_context
                print(f"📱 [앱 컨텍스트] {question_type.upper()} 의도 감지 - 설치된 앱 정보를 질문에 추가했습니다")

        context, source_files, related_concepts, concepts = "", [], [], []
        intent_info = None

        if handler.rag_system:
            result = await handler.get_related_documents(request)
            if len(result) >= 6:
                context, source_files, related_concepts, concepts, intent_info = result[:5]
            elif len(result) >= 5:
                context, source_files, related_concepts, concepts, intent_info = result
            else:
                context, source_files, related_concepts, concepts = result

        if response_mode == 'complete':
            logger.info(f"📄 {handler.__class__.__name__} 완전한 응답 모드")
            complete_result = await handler.generate_complete(
                request, user_id, channel_id, connection, thread_ts,
                context, source_files, related_concepts, concepts, intent_info
            )
            return {
                "answer": complete_result["answer"],
                "intent": complete_result.get("intent"),
                "context_sources": complete_result.get("sources", []),
                "response_time": complete_result.get("response_time", 0)
            }
        elif response_mode == 'streaming':
            logger.info(f"📡 {handler.__class__.__name__} 스트리밍 모드")
            return {
                "answer_stream": handler.generate_stream(
                    request, user_id, channel_id, connection, thread_ts,
                    context, source_files, related_concepts, concepts, intent_info
                ),
                "context_sources": source_files
            }
        else:
            logger.warning(f"알 수 없는 응답 모드: {response_mode}, 스트리밍으로 fallback")
            return {
                "answer_stream": handler.generate_stream(
                    request, user_id, channel_id, connection, thread_ts,
                    context, source_files, related_concepts, concepts, intent_info
                ),
                "context_sources": source_files
            }

    async def process_question_error(self, request: Dict, user_id=None, channel_id=None, connection=None, thread_ts=None) -> Dict:
        try:
            h = self.handlers.get(self.model)
            if h and h.available:
                return await h.process_question_error(request, user_id, channel_id, connection, thread_ts)
            else:
                async def error_stream():
                    yield "사용 가능한 AI 모델이 없습니다."

                return {
                    "answer_stream": error_stream(),
                    "context_sources": []
                }

        except Exception as e:
            print(f"LLMHandler process_question_error 오류: {str(e)}")

            async def error_stream():
                yield f"에러 처리 중 오류가 발생했습니다: {str(e)}"

            return {
                "answer_stream": error_stream(),
                "context_sources": []
            }

    def get_intent_analysis_status(self) -> Dict[str, Any]:
        try:
            if not self.intent_analyzer:
                return {
                    "intent_analyzer_enabled": False,
                    "intent_analyzer_available": False,
                    "llm_connected": False,
                    "analysis_mode": "none"
                }

            return {
                "intent_analyzer_enabled": True,
                "intent_analyzer_available": self.intent_analyzer.is_llm_connected(),
                "llm_connected": self.intent_analyzer.is_llm_connected(),
                "analysis_mode": self.intent_analyzer.get_analysis_mode() if self.intent_analyzer.is_llm_connected() else "휴리스틱"
            }

        except Exception as e:
            print(f"의도 분석 상태 확인 오류: {e}")
            return {
                "intent_analyzer_enabled": bool(self.intent_analyzer),
                "intent_analyzer_available": False,
                "llm_connected": False,
                "analysis_mode": "error"
            }

    def get_chat_session(self, user_id: str, channel_id: str = None) -> Any:
        try:
            session_key = f"{user_id}_{channel_id}" if channel_id else user_id

            if session_key not in self.chat_sessions:
                from aibot_llm_module import ChatSession
                self.chat_sessions[session_key] = ChatSession(session_id=session_key, max_messages=10)

            return self.chat_sessions[session_key]

        except Exception as e:
            print(f"채팅 세션 생성 오류: {e}")
            h = self.handlers.get(self.model)
            if h:
                return h.chat_session
            from aibot_llm_module import ChatSession
            return ChatSession()

    def clean_old_sessions(self, max_age_seconds: int = 3600):
        try:
            if hasattr(self, 'chat_sessions') and self.chat_sessions:
                current_time = time.time()
                old_sessions = [
                    sid for sid, session in self.chat_sessions.items()
                    if current_time - getattr(session, 'last_activity', 0) > max_age_seconds
                ]

                for session_id in old_sessions:
                    del self.chat_sessions[session_id]

                if old_sessions:
                    print(f"🧹 {len(old_sessions)}개 오래된 세션 정리 완료 (남은 세션: {len(self.chat_sessions)}개)")

                return len(old_sessions)

            return 0

        except Exception as e:
            print(f"⚠️ 세션 정리 중 오류 (무시됨): {e}")
            return 0

    def reload_embedding(self, sub_id=None):
        try:
            if self.rag_system:
                print(f"ID : {id(self.rag_system)}")
                if sub_id:
                    print(f"🔄 RAG 시스템 리소스 리로드 중 (sub_id: {sub_id})...")
                else:
                    print("🔄 RAG 시스템 리소스 리로드 중...")
                self.rag_system.reload_resources(sub_id=sub_id)
                print("✅ RAG 시스템 리로드 완료")
        except Exception as e:
            print(f"⚠️ 임베딩 리로드 중 오류: {e}")

    def _parse_security_analysis(self, analysis_text: str) -> Optional[SecurityAnalysisResult]:
        try:
            analysis_json = json.loads(analysis_text)

            return SecurityAnalysisResult(
                classification=analysis_json.get("classification", "unknown"),
                analysis=analysis_json.get("analysis", ""),
                updated_queue=analysis_json.get("updated_queue", [])
            )

        except json.JSONDecodeError:
            return SecurityAnalysisResult(
                classification="unknown",
                analysis=analysis_text,
                updated_queue=[]
            )

