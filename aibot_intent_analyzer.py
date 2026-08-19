import json
import re
import asyncio
import os
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import tiktoken

from config_utils import ConfigManager, load_prompt_file

# command_ref 분류 프롬프트 폴백 — OpenAI 경로·GPT-OSS(Harmony) 경로 공용 (한쪽만 고치는 분기 방지)
_COMMAND_REF_FALLBACK = (
    'You are a strict classifier for query language questions. Decide if the user asks '
    'about the usage or syntax of a SPECIFIC command or function. Respond with exactly '
    'one word: COMMAND_REF or GENERAL.')

@dataclass
class IntentScore:
    qna: float
    plan: float
    action: float
    reasoning: str
    
    def get_primary_intent(self) -> str:
        scores = {'qna': self.qna, 'plan': self.plan, 'action': self.action}
        return max(scores.keys(), key=lambda k: scores[k])
    
    def get_sorted_intents(self) -> List[Tuple[str, float]]:
        scores = {'qna': self.qna, 'plan': self.plan, 'action': self.action}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def is_multi_intent(self, threshold: float = 0.30) -> bool:
        scores = [self.qna, self.plan, self.action]
        sorted_scores = sorted(scores, reverse=True)
        
        return (sorted_scores[1] >= threshold and 
                sorted_scores[0] - sorted_scores[1] <= 0.3)
    
    def get_intent_distribution(self) -> str:
        return f"QNA:{self.qna:.2f} | PLAN:{self.plan:.2f} | ACTION:{self.action:.2f}"
    
    def __str__(self) -> str:
        primary = self.get_primary_intent().upper()
        multi = " (복합)" if self.is_multi_intent() else ""
        return f"IntentScore({self.get_intent_distribution()}) -> {primary}{multi}"

class IntentAnalyzer:
    def __init__(self, llm_client=None, config_manager=None):
        try:
            self.llm_client = llm_client
            
            if config_manager is None:
                self.config_manager = ConfigManager()
            else:
                self.config_manager = config_manager
            
            self.intent_prompt = self._load_intent_prompt()
            self.qna_subtype_prompt = self._load_qna_subtype_prompt()
            
            self._initialize_pattern_matchers()
            
            print(f"✅ 의도 분석기 초기화 완료")
            print(f"   LLM 클라이언트: {'있음' if llm_client else '없음 (휴리스틱 모드)'}")
            print(f"   메인 프롬프트: {'성공' if self.intent_prompt else '실패'}")
            print(f"   QNA 프롬프트: {'성공' if self.qna_subtype_prompt else '실패'}")
            
        except Exception as e:
            print(f"⚠️ 의도 분석기 초기화 중 오류: {e}")
            self.llm_client = llm_client
            self.config_manager = config_manager or ConfigManager()
            self.intent_prompt = self._get_default_intent_prompt()
            self.qna_subtype_prompt = self._get_default_qna_subtype_prompt()
            self._initialize_pattern_matchers()
            print("✅ 의도 분석기 기본 모드로 초기화 완료")

    def _initialize_pattern_matchers(self):
        self.clear_qna_patterns = [
            r'.*무엇인가요\?',
            r'.*뭐야\?',
            r'.*뭔가요\?',
            r'.*설명해(주세요|줘)',
            r'.*알려주세요',
            r'.*차이.*무엇',
            r'.*차이점.*뭔가요',
            r'.*의미.*무엇',
            r'.*개념.*무엇',
            r'.*란\s+(무엇|뭐)',
            r'.*어떻게\s+(해야|하는).*하나요\?',
            r'.*방법.*알려주세요',
            r'.*이해하고\s+싶',
            r'.*궁금합니다'
        ]
        
        self.clear_action_patterns = [
            r'.*찾아줘\s*$',
            r'.*보여줘\s*$',
            r'.*만들어줘\s*$',
            r'.*실행해줘\s*$',
            r'.*가져와줘\s*$',
            r'.*조회해줘\s*$',
            r'.*확인해줘\s*$',
            r'.*세어줘\s*$',
            r'.*계산해줘\s*$',
            r'.*추출해줘\s*$',
            r'.*상위\s*\d+.*찾아',
            r'.*최근\s*\d+.*보여',
            r'.*현황.*보여',
            r'.*통계.*확인',
            r'.*추이.*분석',
            r'.*목록.*조회',
            r'.*개수.*세어',
            r'.*합계.*구해',
            r'.*비율.*계산'
        ]

        self.clear_plan_patterns = [
            r'.*분석하고.*제안',
            r'.*조사하고.*대응', 
            r'.*찾고.*보고서',
            r'.*수집하고.*분석하고.*탐지',
            r'.*단계별로.*실행',
            r'.*순서대로.*진행',
            r'.*과정을.*실행',
            r'.*부터.*까지.*전체.*과정',
            r'.*종합적으로.*분석.*후.*제안',
        ]
        
        self.clear_patterns = {
            'qna': self.clear_qna_patterns,
            'action': self.clear_action_patterns,
            'plan': self.clear_plan_patterns
        }

    def _load_intent_prompt(self) -> str:
        try:
            try:
                prompt_config = self.config_manager.get_prompt_config()
                intent_prompt_path = prompt_config.get('intent')
            except (AttributeError, KeyError) as e:
                print(f"⚠️ config에서 intent 프롬프트 경로를 찾을 수 없습니다: {e}")
                return self._get_default_intent_prompt()
            
            if not intent_prompt_path or not os.path.exists(intent_prompt_path):
                print(f"⚠️ 의도 분석 프롬프트 파일이 없습니다: {intent_prompt_path}")
                return self._get_default_intent_prompt()
            
            file_ext = os.path.splitext(intent_prompt_path)[1].lower()
            
            if file_ext in ['.yaml', '.yml']:
                import yaml
                with open(intent_prompt_path, 'r', encoding='utf-8') as f:
                    prompt_yaml = yaml.safe_load(f)
                    
                    if 'prompt' not in prompt_yaml:
                        print(f"⚠️ YAML 파일에 'prompt' 필드가 없습니다")
                        return self._get_default_intent_prompt()
                    
                    print(f"✅ 의도 분석 프롬프트를 YAML에서 로드")
                    return prompt_yaml['prompt']
            else:
                with open(intent_prompt_path, 'r', encoding='utf-8') as f:
                    prompt = f.read()
                    print(f"✅ 의도 분석 프롬프트를 텍스트에서 로드")
                    return prompt
                    
        except Exception as e:
            print(f"❌ 의도 분석 프롬프트 로드 중 오류: {str(e)}")
            return self._get_default_intent_prompt()

    def _load_qna_subtype_prompt(self) -> str:
        return self._get_default_qna_subtype_prompt()

    def _get_default_intent_prompt(self) -> str:
        prompt = """당신은 사용자의 질문을 분석하여 다음 3가지 의도로 정확히 분류하는 전문가입니다.

                ## 🎯 의도 정의 (명확한 구분 기준)

                ### 1. QNA (질문/답변): 0.0~1.0
                **목적**: 정보 습득, 개념 이해, 설명 요청
                **핵심 기준**: "알고 싶다" + "실행은 불필요"

                **QNA 신호 단어:**
                - 질문형: "무엇인가요?", "어떤", "왜", "어떻게 해야"
                - 설명 요청: "설명해주세요", "알려주세요", "이해하고 싶어요"
                - 정의 요청: "란 무엇", "의미", "개념", "차이점"
                - 비교 요청: "차이", "장단점", "vs", "비교"

                ### 2. ACTION (실행/작업): 0.0~1.0
                **목적**: 단일 쿼리/스크립트로 완료 가능한 즉시 실행 작업
                **핵심 기준**: "하나의 작업으로 완료 가능" + "즉시 실행"

                **ACTION 신호 단어:**
                - 즉시 실행: "찾아줘", "보여줘", "만들어줘", "실행해줘"
                - 단일 결과: "쿼리 만들어", "현황", "통계", "목록", "추이"
                - 명확한 범위: "상위 N개", "최근 기간", "특정 조건"

                ### 3. PLAN (계획/단계): 0.0~1.0
                **목적**: 여러 쿼리/단계가 필요한 복합 작업, 순차적 실행
                **핵심 기준**: "여러 단계 필요" + "복합 분석" + "순차 실행"

                **PLAN 신호 단어:**
                - 다단계: "단계별로", "순서대로", "과정을", "절차"
                - 복합 작업: "분석하고 + 제안", "찾고 + 보고서", "조사 + 대응"
                - 전체적 접근: "종합적으로", "전체적으로", "통합", "전반적"
                - 연결어: "그리고", "다음에", "이후에", "부터~까지"

                응답 형식 (JSON):
                [{"qna": 0.0, "plan": 0.0, "action": 1.0, "reasoning": "분석 근거"}]"""
        return prompt

    def _get_default_qna_subtype_prompt(self) -> str:
        """QNA 하위 분류 기본 프롬프트.

        라벨 'CVE'/'GENERAL' 은 원본에서 물려받은 **토큰 이름**이다(파싱부가 이 문자열을 비교한다).
        의미는 도메인 중립이다 — CVE = 참조 문서의 특정 항목을 짚어야 하는 질문,
        GENERAL = 문서 항목을 짚지 않는 일반 질문. 카트리지가 자기 도메인 어휘로 덮어쓴다.
        """
        prompt = """당신은 질문의 의도를 정확히 파악하는 분류 전문가입니다.
                질문을 [CVE] 또는 [GENERAL] 중 하나로 분류하세요.

                📌 분류 기준:

                [CVE] — 참조 문서의 **특정 항목**을 찾아야 답할 수 있는 질문:
                • 식별자(제품명·모델명·항목 번호·규격 번호 등)가 포함된 경우
                • "이 항목이 무엇인가", "여기 해당하는가", "어떤 종류인가" 판단 요청
                • 특정 대상 + 속성을 묻는 경우 (해당 문서를 찾아 근거로 답해야 함)
                • 구체적 문자열·경로·코드 조각을 제시하며 그것이 무엇인지 묻는 경우

                [GENERAL] — 특정 문서 항목을 짚지 않는 일반 질문:
                • 개념·용어 설명 요청
                • 도구/기능의 일반적인 사용법
                • 데이터 변환·질의 생성 등 처리 요청
                • 절차나 방법에 대한 포괄적 질문

                ---
                예시:
                "MODEL-2024-113 사양 알려줘" → CVE
                "이 경로 /api/v1/items 는 뭐하는 건가요?" → CVE
                "A제품 3.2 버전 제한사항" → CVE
                "인덱스가 뭔가요?" → GENERAL
                "로그 파싱 쿼리 만들어줘" → GENERAL
                "이 도구 사용법 설명해줘" → GENERAL

                ---
                질문: {query}

                한 단어로만 답하세요: CVE 또는 GENERAL
                분류:"""
        return prompt

    def connect_llm(self, llm_handler):
        if llm_handler is not None:
            self.llm_client = llm_handler
            print("🔗 의도 분석기 <- LLM 연결 완료")
            print(f"   연결된 LLM: {type(llm_handler).__name__}")
            return True
        else:
            print("⚠️ LLM 핸들러 연결 실패: None 객체")
            return False

    def is_llm_connected(self) -> bool:
        return self.llm_client is not None

    def _is_gptoss_client(self, client) -> bool:
        if not client:
            return False
        
        if 'gptoss' in client.__class__.__name__.lower():
            print(f"🔍 GPT-OSS 감지: 클래스명 '{client.__class__.__name__}'")
            return True
        
        model_name = getattr(client, 'model_name', '').lower()
        if 'gpt-oss' in model_name:
            print(f"🔍 GPT-OSS 감지: model_name '{model_name}'")
            return True
        
        if hasattr(client, 'local_model') and client.local_model:
            if hasattr(client.local_model, 'model_path'):
                model_path = getattr(client.local_model, 'model_path', '').lower()
                if 'gpt-oss' in model_path:
                    print(f"🔍 GPT-OSS 감지: model_path '{model_path}'")
                    return True
            
            if hasattr(client.local_model, '__class__'):
                local_model_class = client.local_model.__class__.__name__.lower()
                if 'gptoss' in local_model_class:
                    print(f"🔍 GPT-OSS 감지: local_model 클래스 '{local_model_class}'")
                    return True
        
        if hasattr(client, 'gptoss_handler'):
            print(f"🔍 GPT-OSS 감지: gptoss_handler 속성 존재")
            return True
        
        return False

    def get_analysis_mode(self) -> str:
        if self.is_llm_connected():
            if self._is_gptoss_client(self.llm_client):
                return "GPT-OSS (로컬)"
            elif hasattr(self.llm_client, 'chat'):
                return "OpenAI"
            elif hasattr(self.llm_client, 'local_model'):
                return "LLaMA (로컬)"
            elif hasattr(self.llm_client, 'bedrock_runtime'):
                return "LLaMA (AWS Bedrock)"
            else:
                return "LLM (일반)"
        else:
            return "휴리스틱"

    def _create_gptoss_qna_prompt(self, query: str) -> str:
        # 라벨 'CVE'/'GENERAL' 은 파싱부가 비교하는 토큰 이름이다 (위 _get_default_qna_subtype_prompt 주석 참조).
        system_content = """
You are a strict classifier.
Decide whether a question needs a **specific item from the reference documents** (CVE)
or is a **general question** that does not point at one (GENERAL).

Rules:

- CVE:
  - The question contains an identifier (product name, model number, item code, spec number, path, or code fragment)
    **and asks what it is**, **whether it applies**, or requests **analysis or explanation of that specific thing**
  - The question names a specific target and asks about its attributes, so answering requires locating the matching document
  - The question quotes a concrete string, path, or snippet and asks what it means

- GENERAL:
  - The question is about concepts, terminology, tools, formats, or procedures **without pointing at a specific documented item**
  - It asks for **data transformation**, **query or schema generation**, or other general processing
  - It includes a path or an odd-looking string, but the intent is to **understand** or **transform** it, not to look it up

🚨 FINAL RULE: Your response MUST be exactly one word — either 'CVE' or 'GENERAL'.
No other words. No explanation. No punctuation.

Output exactly:
<|start|>assistant<|channel|>final<|message|>CVE
        """

        chatml_prompt = f"""<|im_start|>system
                            {system_content}
                            <|im_end|>
                            <|im_start|>user
                            Determine whether the query '{query}' should be classified as CVE or GENERAL.
                            <|im_end|>
                            <|im_start|>assistant
                        """
        return chatml_prompt

    def _clean_gptoss_response(self, response: str, is_qna: bool = False) -> str:
        if "<|message|>" in response:
            message_start = response.find("<|message|>") + len("<|message|>")
            response = response[message_start:].strip()
        
        cleanup_tokens = ["<|im_end|>", "<|endoftext|>", "<|end|>", "<|im_start|>"]
        for token in cleanup_tokens:
            response = response.replace(token, "")
        
        response = response.strip()
        
        if is_qna:
            response_upper = response.upper()
            if "CVE" in response_upper and "GENERAL" not in response_upper:
                response = "CVE"
            elif "GENERAL" in response_upper and "CVE" not in response_upper:
                response = "GENERAL"
            elif "GENERAL" in response_upper:  
                response = "GENERAL"
            else:
                first_word = response.split()[0] if response.split() else "GENERAL"
                response = "CVE" if "CVE" in first_word.upper() else "GENERAL"
        else:
            if not response.startswith('['):
                response = '[' + response
        
        return response
        
    async def analyze_intent(self, query: str) -> IntentScore:
        try:
            mode = self.get_analysis_mode()
            print(f"🧠 의도 분석 모드: {mode}")
            
            if self.is_llm_connected():
                print(f"🔍 LLM 클라이언트 타입: {type(self.llm_client).__name__}")
                print(f"🔍 LLM 클라이언트 속성 체크:")
                print(f"   - has chat: {hasattr(self.llm_client, 'chat')}")
                print(f"   - has local_model: {hasattr(self.llm_client, 'local_model')}")
                print(f"   - has bedrock_runtime: {hasattr(self.llm_client, 'bedrock_runtime')}")
                print(f"   - model_name: {getattr(self.llm_client, 'model_name', 'None')}")
                print(f"   - handler_type: {getattr(self.llm_client, 'handler_type', 'None')}")
                print(f"   - is_gptoss: {self._is_gptoss_client(self.llm_client)}")
            
            if self.is_llm_connected():
                print(f"🤖 {mode} 모드로 의도 분석 수행")
                try:
                    intent_data = await self._perform_llm_analysis(query)
                    intent_data = self._post_process_scores(query, intent_data)
                    
                    return IntentScore(
                        qna=intent_data['qna'],
                        plan=intent_data['plan'], 
                        action=intent_data['action'],
                        reasoning=f"[{mode}] {intent_data['reasoning']}"
                    )
                except Exception as e:
                    print(f"⚠️ LLM 분석 실패: {e}")
            
            print(f"🔄 휴리스틱 분석으로 전환")
            return self._fallback_intent_analysis(query)
            
        except Exception as e:
            print(f"⚠️ 의도 분석 중 오류 ({self.get_analysis_mode()}): {str(e)}")
            return self._fallback_intent_analysis(query)

    async def analyze_qna_subtype(self, query: str) -> str:
        try:
            if not self.is_llm_connected():
                print("⚠️ QNA 세부 분석은 LLM이 필요합니다. 휴리스틱 사용 불가")
                return "GENERAL"
            
            mode = self.get_analysis_mode()
            print(f"🔍 QNA 세부 분석 모드: {mode}")
            
            result = await self._perform_qna_subtype_analysis(query)
            
            if isinstance(result, str):
                clean_result = result.strip().upper()
                if "CVE" in clean_result:
                    return "CVE"
                elif "GENERAL" in clean_result:
                    return "GENERAL"
                else:
                    print(f"⚠️ 예상하지 못한 응답: {result}, GENERAL로 기본값 설정")
                    return "GENERAL"
            else:
                print(f"⚠️ 잘못된 응답 형식: {type(result)}, GENERAL로 기본값 설정")
                return "GENERAL"
            
        except Exception as e:
            print(f"⚠️ QNA 세부 분석 중 오류: {str(e)}")
            return "GENERAL"

    def analyze_qna_subtype_sync(self, query: str) -> str:
        try:
            if not self.is_llm_connected():
                print("⚠️ QNA 세부 분석은 LLM이 필요합니다. 휴리스틱 사용 불가")
                return "GENERAL"
            
            try:
                loop = asyncio.get_running_loop()
                print("🔍 이벤트 루프 실행 중 - 스레드 기반 실행 사용")
                
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self.analyze_qna_subtype(query))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    return future.result(timeout=10)
                    
            except RuntimeError:
                print("🔍 이벤트 루프 없음 - 직접 실행")
                return asyncio.run(self.analyze_qna_subtype(query))
                
        except Exception as e:
            print(f"⚠️ QNA 세부 분석 동기 실행 중 오류: {str(e)}")
            return "GENERAL"

    def analyze_command_ref_sync(self, query: str) -> str:
        """
        번역된 질문이 특정 명령어/함수의 사용법을 묻는 것인지 분류.
        Returns: 'COMMAND_REF' 또는 'GENERAL'
        """
        try:
            if not self.is_llm_connected():
                print("⚠️ 명령어 의도 분류는 LLM이 필요합니다")
                return "GENERAL"

            prompt = self._create_gptoss_command_ref_prompt(query)

            # LLM 클라이언트 분기
            _openai_client = getattr(self.llm_client, 'client', self.llm_client)
            _default_model = self.config_manager.get_model_config().get('model_name', 'gpt-4o-mini')
            if hasattr(self.llm_client, 'local_model') and self.llm_client.local_model:
                response = self.llm_client.local_model.generate_complete(
                    prompt=prompt,
                    max_tokens=20,
                    temperature=0,
                    top_p=0.8,
                )
            elif hasattr(_openai_client, 'chat') and hasattr(_openai_client.chat, 'completions'):
                # OpenAI API 직접 호출 (동기) - Harmony 토큰 제거, messages 형식으로 변환
                _model = getattr(self.llm_client, 'model', _default_model)
                # 분류 프롬프트는 카트리지가 정의 — [prompts] command_ref_classifier
                system_msg = load_prompt_file('command_ref_classifier', _COMMAND_REF_FALLBACK)
                # 단순 분류는 경량 모델 사용 (속도 최적화)
                _light_model = "gpt-4o-mini"
                result = _openai_client.chat.completions.create(
                    model=_light_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": f"Classify: {query}"}
                    ],
                    max_completion_tokens=20,
                )
                response = result.choices[0].message.content
            else:
                print("⚠️ 지원되지 않는 LLM 클라이언트")
                return "GENERAL"

            # 응답 정리
            cleaned = self._clean_gptoss_response(response, is_qna=False)
            cleaned_upper = cleaned.upper().strip()

            if "COMMAND_REF" in cleaned_upper:
                print(f"🔍 [명령어 의도] COMMAND_REF 판정: '{query[:50]}'")
                return "COMMAND_REF"
            else:
                print(f"🔍 [명령어 의도] GENERAL 판정: '{query[:50]}'")
                return "GENERAL"

        except Exception as e:
            print(f"⚠️ 명령어 의도 분류 오류: {e}")
            return "GENERAL"

    def _create_gptoss_command_ref_prompt(self, query: str) -> str:
        """명령어/함수 사용법 질문 여부를 분류하는 GPT-OSS 프롬프트 (Harmony 형식)"""
        _cls = load_prompt_file('command_ref_classifier', _COMMAND_REF_FALLBACK)
        return (
            f"<|start|>system\n{_cls}\n<|end|>\n"
            f"<|start|>user\nClassify: {query}\n<|end|>\n"
            f"<|start|>assistant<|channel|>final<|message|>"
        )

    async def _perform_qna_subtype_analysis(self, query: str) -> str:
        _oc = getattr(self.llm_client, 'client', self.llm_client)
        if hasattr(_oc, 'chat') and hasattr(_oc.chat, 'completions'):
            response = await self._analyze_qna_with_openai(query)
            return response
        elif self._is_gptoss_client(self.llm_client):
            response = await self._analyze_qna_with_gptoss_handler(query)
            return response
        elif (hasattr(self.llm_client, 'bedrock_runtime') or 
            hasattr(self.llm_client, 'local_model') or
            hasattr(self.llm_client, 'model_id') or
            hasattr(self.llm_client, '_initialize_model')):  
            response = await self._analyze_qna_with_llama_handler(query)
            return response
        else: 
            response = await self._analyze_qna_with_generic_llm(query)
            return response

    async def _analyze_qna_with_gptoss_handler(self, query: str) -> str:
        try:
            formatted_prompt = self._create_gptoss_qna_prompt(query)
            
            if hasattr(self.llm_client, 'local_model') and self.llm_client.local_model:
                response = await self._call_local_model_for_gptoss_qna(formatted_prompt)
            else:
                response = await self._call_generic_gptoss_for_qna(formatted_prompt)

            return self._clean_gptoss_response(response, is_qna=True)
            
        except Exception as e:
            print(f"GPT-OSS QNA 분석 오류: {str(e)}")
            raise

    async def _call_local_model_for_gptoss_qna(self, prompt: str) -> str:
        try:
            return self.llm_client.local_model.generate_complete(
                prompt=prompt,
                max_tokens=1000,
                temperature=0,
                top_p=0.8,
            )
        except Exception as e:
            print(f"GPT-OSS QNA 분석 실패: {e}")
            raise

    async def _call_generic_gptoss_for_qna(self, prompt: str) -> str:
        try:
            if hasattr(self.llm_client, 'generate_complete'):
                if asyncio.iscoroutinefunction(self.llm_client.generate_complete):
                    return await self.llm_client.generate_complete(
                        prompt=prompt,
                        max_tokens=50,
                        temperature=0.1
                    )
                else:
                    return self.llm_client.generate_complete(
                        prompt=prompt,
                        max_tokens=50,
                        temperature=0.1
                    )
            else:
                raise NotImplementedError("지원되지 않는 GPT-OSS 클라이언트 인터페이스")
        except Exception as e:
            print(f"일반 GPT-OSS QNA 분석 실패: {e}")
            raise

    async def _perform_llm_analysis(self, query: str) -> Dict[str, Union[float, str]]:
        _oc = getattr(self.llm_client, 'client', self.llm_client)
        if hasattr(_oc, 'chat') and hasattr(_oc.chat, 'completions'):
            response = await self._analyze_with_openai(query)
            return self._parse_intent_response(response, llm_type="openai")
        elif (hasattr(self.llm_client, 'bedrock_runtime') or 
            hasattr(self.llm_client, 'local_model') or
            hasattr(self.llm_client, 'model_id') or
            hasattr(self.llm_client, '_initialize_model')):  
            response = await self._analyze_with_llama_handler(query)
            return self._parse_intent_response(response, llm_type="llama")
        elif self._is_gptoss_client(self.llm_client):  # 🆕 GPT-OSS 감지
            response = await self._analyze_with_gptoss_handler(query)
            return self._parse_intent_response(response, llm_type="gpt-oss")
        else:  
            response = await self._analyze_with_generic_llm(query)
            return self._parse_intent_response(response, llm_type="generic")

    async def _analyze_with_gptoss_handler(self, query: str) -> str:
        try:
            formatted_prompt = self._create_gptoss_intent_prompt(query)
            
            if hasattr(self.llm_client, 'local_model') and self.llm_client.local_model:
                response = await self._call_local_model_via_gptoss_handler(formatted_prompt)
            else:
                response = await self._call_generic_gptoss_handler(formatted_prompt)
            
            return self._clean_gptoss_response(response)
            
        except Exception as e:
            print(f"GPT-OSS 핸들러 호출 오류: {str(e)}")
            raise

    async def _call_local_model_via_gptoss_handler(self, prompt: str) -> str:
        try:
            return self.llm_client.local_model.generate_complete(
                prompt=prompt,
                max_tokens=200,
                temperature=0.05,
                top_p=0.8
            )
        except Exception as e:
            print(f"GPT-OSS 로컬 모델 호출 실패: {e}")
            raise

    async def _call_generic_gptoss_handler(self, prompt: str) -> str:
        try:
            if hasattr(self.llm_client, 'generate_complete'):
                if asyncio.iscoroutinefunction(self.llm_client.generate_complete):
                    return await self.llm_client.generate_complete(
                        prompt=prompt,
                        max_tokens=200,
                        temperature=0.1
                    )
                else:
                    return self.llm_client.generate_complete(
                        prompt=prompt,
                        max_tokens=200,
                        temperature=0.1
                    )
            else:
                raise NotImplementedError("지원되지 않는 GPT-OSS 클라이언트 인터페이스")
        except Exception as e:
            print(f"일반 GPT-OSS 호출 실패: {e}")
            raise

    async def _analyze_with_openai(self, query: str) -> str:
        try:
            messages = [
                {"role": "system", "content": self.intent_prompt},
                {"role": "user", "content": query}
            ]

            client = getattr(self.llm_client, 'client', self.llm_client)
            _default_model = self.config_manager.get_model_config().get('model_name', 'gpt-4o-mini')
            model = getattr(self.llm_client, 'model', _default_model)
            if hasattr(client.chat.completions, 'acreate'):
                response = await client.chat.completions.acreate(
                    model=model,
                    messages=messages,
                    max_completion_tokens=200
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=200
                )

            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API 호출 오류: {str(e)}")
            raise

    async def _analyze_with_llama_handler(self, query: str) -> str:
        try:
            formatted_prompt = f"""<|start_header_id|>system<|end_header_id|>
{self.intent_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
["""
            
            if hasattr(self.llm_client, 'local_model') and self.llm_client.local_model:
                response = await self._call_local_model_via_handler(formatted_prompt)
            elif hasattr(self.llm_client, 'bedrock_runtime'):
                response = await self._call_bedrock_via_handler(formatted_prompt)
            else:
                response = await self._call_generic_llama(formatted_prompt)
            
            if not response.strip().startswith('['):
                response = '[' + response
            
            return response
        except Exception as e:
            print(f"LLaMA 핸들러 호출 오류: {str(e)}")
            raise

    async def _call_local_model_via_handler(self, prompt: str) -> str:
        try:
            return self.llm_client.local_model.generate_complete(
                prompt=prompt,
                max_tokens=200,
                temperature=0.05,
                top_p=0.8
            )
        except Exception as e:
            print(f"로컬 모델 호출 실패: {e}")
            raise

    async def _call_bedrock_via_handler(self, prompt: str) -> str:
        try:
            import json
            
            request_body = {
                "prompt": prompt,
                "max_gen_len": 200,
                "temperature": 0.1,
                "top_p": 0.9
            }
            
            response = self.llm_client.bedrock_runtime.invoke_model(
                modelId=self.llm_client.model_id,
                body=json.dumps(request_body),
                contentType="application/json"
            )
            response_body = json.loads(response['body'].read())
            return response_body.get('generation', '')
        except Exception as e:
            print(f"Bedrock 호출 실패: {e}")
            raise

    async def _call_generic_llama(self, prompt: str) -> str:
        try:
            if hasattr(self.llm_client, 'generate_complete'):
                if asyncio.iscoroutinefunction(self.llm_client.generate_complete):
                    return await self.llm_client.generate_complete(
                        prompt=prompt,
                        max_tokens=200,
                        temperature=0.1
                    )
                else:
                    return self.llm_client.generate_complete(
                        prompt=prompt,
                        max_tokens=200,
                        temperature=0.1
                    )
            else:
                raise NotImplementedError("지원되지 않는 LLaMA 클라이언트 인터페이스")
        except Exception as e:
            print(f"일반 LLaMA 호출 실패: {e}")
            raise

    async def _analyze_with_generic_llm(self, query: str) -> str:
        if hasattr(self.llm_client, 'generate'):
            return self.llm_client.generate(
                prompt=f"{self.intent_prompt}\n\nUser: {query}\nAssistant:",
                max_tokens=200,
                temperature=0.1
            )
        else:
            raise NotImplementedError("지원되지 않는 LLM 클라이언트 인터페이스")

    def _parse_intent_response(self, response: str, llm_type: str = "generic") -> Dict[str, Union[float, str]]:
        try:
            response = response.strip()
            
            if llm_type == "llama":
                return self._parse_llama_response(response)
            elif llm_type == "openai":
                return self._parse_openai_response(response)
            elif llm_type == "gpt-oss": 
                return self._parse_gptoss_response(response)
            else:
                return self._parse_generic_response(response)
            
        except Exception as e:
            print(f"⚠️ {llm_type} 응답 파싱 중 예외: {str(e)}")
            return self._fallback_scoring_for_llm(llm_type, response)

    def _parse_gptoss_response(self, response: str) -> Dict[str, Union[float, str]]:
        try:
            if "<|message|>" in response:
                message_start = response.find("<|message|>") + len("<|message|>")
                response = response[message_start:].strip()
            
            cleanup_tokens = ["<|im_end|>", "<|endoftext|>", "<|end|>"]
            for token in cleanup_tokens:
                response = response.replace(token, "")
            
            json_patterns = [
                r'\[\s*\{[^}]*"qna"[^}]*\}\s*\]',
                r'\{[^{}]*"qna"[^{}]*\}',
                r'```json\s*(\[[^]]*\])\s*```',
                r'```json\s*(\{[^}]*\})\s*```',
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    json_str = match.group(1) if match.groups() else match.group(0)
                    break
            
            if not json_str:
                return self._extract_scores_aggressive(response, "gpt-oss")
            
            try:
                if json_str.strip().startswith('['):
                    parsed = json.loads(json_str)
                    intent_data = parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else {}
                else:
                    intent_data = json.loads(json_str)
            except json.JSONDecodeError:
                return self._extract_scores_aggressive(response, "gpt-oss")
            
            return self._validate_and_clean_intent_data(intent_data, "GPT-OSS")
            
        except Exception as e:
            print(f"⚠️ GPT-OSS 파싱 오류: {str(e)}")
            return self._extract_scores_aggressive(response, "gpt-oss")

    def _parse_openai_response(self, response: str) -> Dict[str, Union[float, str]]:
        try:
            json_patterns = [
                r'\[\s*\{[^}]*"qna"[^}]*\}\s*\]',
                r'\{[^{}]*"qna"[^{}]*\}',
                r'```json\s*(\[[^]]*\])\s*```',
                r'```json\s*(\{[^}]*\})\s*```',
                r'```\s*(\[[^]]*\])\s*```',
                r'```\s*(\{[^}]*\})\s*```',
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    json_str = match.group(1) if match.groups() else match.group(0)
                    break
            
            if not json_str:
                return self._extract_scores_aggressive(response, "openai")
            
            try:
                if json_str.strip().startswith('['):
                    parsed = json.loads(json_str)
                    intent_data = parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else {}
                else:
                    intent_data = json.loads(json_str)
            except json.JSONDecodeError:
                return self._extract_scores_aggressive(response, "openai")
            
            return self._validate_and_clean_intent_data(intent_data, "OpenAI")
            
        except Exception as e:
            print(f"⚠️ OpenAI 파싱 오류: {str(e)}")
            return self._extract_scores_aggressive(response, "openai")

    def _parse_llama_response(self, response: str) -> Dict[str, Union[float, str]]:
        try:
            if not ('[' in response or '{' in response):
                return self._analyze_llama_natural_response(response)
            
            json_patterns = [
                r'\[\s*\{[^}]*"qna"[^}]*\}\s*\]',
                r'\{[^{}]*"qna"[^{}]*"plan"[^{}]*"action"[^{}]*\}',
                r'(\{[^{}]*(?:"qna"|"plan"|"action")[^{}]*\})',
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
                if match:
                    json_str = match.group(1) if match.groups() else match.group(0)
                    break
            
            if not json_str:
                return self._extract_scores_aggressive(response, "llama")
            
            try:
                if json_str.strip().startswith('['):
                    parsed = json.loads(json_str)
                    intent_data = parsed[0] if isinstance(parsed, list) and len(parsed) > 0 else {}
                else:
                    intent_data = json.loads(json_str)
                
                return self._validate_and_clean_intent_data(intent_data, "Llama")
                
            except json.JSONDecodeError:
                return self._extract_scores_aggressive(response, "llama")
            
        except Exception as e:
            print(f"⚠️ Llama 응답 파싱 중 예외: {str(e)}")
            return self._analyze_llama_natural_response(response)

    def _parse_generic_response(self, response: str) -> Dict[str, Union[float, str]]:
        try:
            return self._parse_openai_response(response)
        except:
            return self._parse_llama_response(response)

    def _validate_and_clean_intent_data(self, intent_data: Dict, model_name: str) -> Dict[str, Union[float, str]]:
        try:
            required_keys = ['qna', 'plan', 'action']
            for key in required_keys:
                if key not in intent_data:
                    intent_data[key] = 0.0
                else:
                    try:
                        intent_data[key] = float(intent_data[key])
                    except (ValueError, TypeError):
                        intent_data[key] = 0.0
            
            if 'reasoning' not in intent_data:
                intent_data['reasoning'] = f"{model_name} 응답에서 추출"
            
            for key in required_keys:
                if intent_data[key] < 0:
                    intent_data[key] = 0.0
                elif intent_data[key] > 1:
                    intent_data[key] = 1.0
            
            total = sum(intent_data[key] for key in required_keys)
            if total <= 0:
                intent_data = {'qna': 0.33, 'plan': 0.33, 'action': 0.34, 'reasoning': f'{model_name} 점수 오류로 기본값'}
            elif abs(total - 1.0) > 0.01:
                for key in required_keys:
                    intent_data[key] /= total
                print(f"📊 {model_name} 점수 정규화: {total:.3f} -> 1.000")
            
            if len(intent_data['reasoning']) > 100:
                intent_data['reasoning'] = intent_data['reasoning'][:97] + "..."
            
            return intent_data
            
        except Exception as e:
            print(f"⚠️ {model_name} 데이터 검증 실패: {e}")
            return {'qna': 0.33, 'plan': 0.33, 'action': 0.34, 'reasoning': f'{model_name} 검증 실패 기본값'}

    def _extract_scores_aggressive(self, response: str, model_type: str) -> Dict[str, Union[float, str]]:
        try:
            score_patterns = [
                r'"?qna"?\s*:\s*(\d+\.?\d*)',
                r'"?plan"?\s*:\s*(\d+\.?\d*)', 
                r'"?action"?\s*:\s*(\d+\.?\d*)',
                r'QNA\s*:\s*(\d+\.?\d*)',
                r'PLAN\s*:\s*(\d+\.?\d*)',
                r'ACTION\s*:\s*(\d+\.?\d*)',
            ]
            
            scores = {'qna': 0.0, 'plan': 0.0, 'action': 0.0}
            
            for i, pattern in enumerate(score_patterns):
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    score = float(match.group(1))
                    intent_name = ['qna', 'plan', 'action'][i % 3]
                    scores[intent_name] = max(scores[intent_name], score)
            
            if any(score > 1.0 for score in scores.values()):
                scores = {k: v/100 for k, v in scores.items()}
            
            if sum(scores.values()) == 0:
                return self._analyze_by_keywords(response, model_type)
            
            total = sum(scores.values())
            if total > 0:
                scores = {k: v/total for k, v in scores.items()}
            
            return {**scores, 'reasoning': f'{model_type} 적극적 추출 결과'}
            
        except Exception as e:
            print(f"⚠️ {model_type} 적극적 추출 실패: {e}")
            return self._analyze_by_keywords(response, model_type)

    def _analyze_llama_natural_response(self, response: str) -> Dict[str, Union[float, str]]:
        return self._analyze_by_keywords(response, "Llama 자연어")

    def _analyze_by_keywords(self, text: str, source: str) -> Dict[str, Union[float, str]]:
        text_lower = text.lower()
        
        qna_keywords = ['정보', '설명', '질문', '답변', '알려', '차이', '의미', 'information', 'explain', 'question']
        plan_keywords = ['계획', '전략', '설계', '단계', '과정', '방법론', 'plan', 'strategy', 'design', 'step']
        action_keywords = ['실행', '생성', '만들기', '찾기', '조회', '작업', 'execute', 'create', 'generate', 'find']
        
        qna_score = sum(1 for kw in qna_keywords if kw in text_lower)
        plan_score = sum(1 for kw in plan_keywords if kw in text_lower)
        action_score = sum(1 for kw in action_keywords if kw in text_lower)
        
        total = qna_score + plan_score + action_score
        if total == 0:
            return {'qna': 0.33, 'plan': 0.33, 'action': 0.34, 'reasoning': f'{source} 키워드 없음, 기본값'}
        
        return {
            'qna': qna_score / total,
            'plan': plan_score / total,
            'action': action_score / total,
            'reasoning': f'{source} 키워드 분석 결과'
        }

    def _fallback_scoring_for_llm(self, llm_type: str, response: str) -> Dict[str, Union[float, str]]:
        if "llama" in llm_type.lower():
            return self._analyze_by_keywords(response, f"{llm_type} fallback")
        elif "gpt-oss" in llm_type.lower():
            return self._analyze_by_keywords(response, f"{llm_type} fallback")
        else:
            return {'qna': 0.1, 'plan': 0.1, 'action': 0.8, 'reasoning': f'{llm_type} 파싱 실패 기본값'}

    def _post_process_scores(self, query: str, scores: Dict) -> Dict:
        query_lower = query.lower()
        original_reasoning = scores.get('reasoning', '')
        
        qna, plan, action = scores['qna'], scores['plan'], scores['action']
        total = qna + plan + action
        
        if total <= 0 or any(score < 0 for score in [qna, plan, action]):
            print("⚠️ 비정상적인 점수 감지, 휴리스틱으로 복구")
            return self._emergency_scoring(query)
        
        if abs(total - 1.0) > 0.01:
            qna, plan, action = qna/total, plan/total, action/total
            print(f"📊 점수 정규화: {total:.3f} -> 1.000")
        
        sorted_scores = sorted([('qna', qna), ('plan', plan), ('action', action)], 
                              key=lambda x: x[1], reverse=True)
        
        max_intent, max_score = sorted_scores[0]
        second_intent, second_score = sorted_scores[1]
        score_gap = max_score - second_score
        
        if score_gap >= 0.4 and max_score >= 0.6:
            if max_intent == 'qna':
                qna = 0.85
                plan = 0.075
                action = 0.075
            elif max_intent == 'plan':
                plan = 0.85
                qna = 0.075
                action = 0.075
            else: 
                action = 0.85
                qna = 0.075
                plan = 0.075
            
            scores['reasoning'] = f'후처리: {max_intent.upper()} 의도 강화 (차이: {score_gap:.2f})'
        
        return {
            'qna': qna,
            'plan': plan,
            'action': action,
            'reasoning': scores.get('reasoning', original_reasoning)
        }

    def _emergency_scoring(self, query: str) -> Dict[str, Union[float, str]]:
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['무엇', '설명', '차이', '의미']):
            return {'qna': 0.8, 'plan': 0.1, 'action': 0.1, 'reasoning': '비상 QNA 점수'}
        elif any(word in query_lower for word in ['찾아', '만들어', '보여', '실행']):
            return {'qna': 0.1, 'plan': 0.1, 'action': 0.8, 'reasoning': '비상 ACTION 점수'}
        else:
            return {'qna': 0.1, 'plan': 0.8, 'action': 0.1, 'reasoning': '비상 PLAN 점수'}

    def _fallback_intent_analysis(self, query: str) -> IntentScore:
        query_lower = query.lower()
        
        qna_keywords = [
            '무엇', '뭐', '어떻게', '왜', '차이', '설명', '의미', '개념', '이란', '란',
            'what', 'how', 'why', 'difference', 'explain', 'meaning', 'concept'
        ]
        
        plan_keywords = [
            '계획', '전략', '방법', '접근', '구조', '설계', '단계별', '순서대로', '과정',
            'plan', 'strategy', 'method', 'approach', 'structure', 'design'
        ]
        
        action_keywords = [
            '찾아', '검색', '보여', '생성', '실행', '가져와', '추출', '분석해', '해줘', '만들어',
            'find', 'search', 'show', 'generate', 'run', 'get', 'extract'
        ]
        
        qna_score = sum(1 for kw in qna_keywords if kw in query_lower)
        plan_score = sum(1 for kw in plan_keywords if kw in query_lower)
        action_score = sum(1 for kw in action_keywords if kw in query_lower)
        
        total = qna_score + plan_score + action_score
        if total == 0:
            return self._determine_default_intent(query)
        
        qna_ratio = qna_score / total
        plan_ratio = plan_score / total
        action_ratio = action_score / total
        
        return IntentScore(
            qna=qna_ratio,
            plan=plan_ratio,
            action=action_ratio,
            reasoning=f'휴리스틱: QNA({qna_score:.1f}), PLAN({plan_score:.1f}), ACTION({action_score:.1f})'
        )

    def _determine_default_intent(self, query: str) -> IntentScore:
        query_lower = query.lower()
        
        if '?' in query:
            return IntentScore(qna=0.8, plan=0.1, action=0.1, 
                            reasoning='질문 형태로 QNA 기본값')
        elif any(query_lower.endswith(word) for word in ['해줘', '줘', '해봐', '실행']):
            return IntentScore(qna=0.1, plan=0.1, action=0.8, 
                            reasoning='명령형으로 ACTION 기본값')
        elif len(query) > 40:
            return IntentScore(qna=0.2, plan=0.6, action=0.2, 
                            reasoning='긴 질문으로 PLAN 기본값')
        else:
            return IntentScore(qna=0.6, plan=0.2, action=0.2, 
                            reasoning='중간 길이로 QNA 기본값')

    async def _analyze_qna_with_openai(self, query: str) -> str:
        try:
            messages = [
                {"role": "system", "content": self.qna_subtype_prompt},
                {"role": "user", "content": query}
            ]

            client = getattr(self.llm_client, 'client', self.llm_client)
            _default_model = self.config_manager.get_model_config().get('model_name', 'gpt-4o-mini')
            model = getattr(self.llm_client, 'model', _default_model)
            if hasattr(client.chat.completions, 'acreate'):
                response = await client.chat.completions.acreate(
                    model=model,
                    messages=messages,
                    max_completion_tokens=200
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=200
                )

            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI QNA 분석 오류: {str(e)}")
            raise

    async def _analyze_qna_with_llama_handler(self, query: str) -> str:
        try:
            formatted_prompt = f"""<|start_header_id|>system<|end_header_id|>
{self.qna_subtype_prompt}
<|eot_id|><|start_header_id|>user<|end_header_id|>
{query}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            
            if hasattr(self.llm_client, 'local_model') and self.llm_client.local_model:
                response = await self._call_local_model_for_qna(formatted_prompt)
            elif hasattr(self.llm_client, 'bedrock_runtime'):
                response = await self._call_bedrock_for_qna(formatted_prompt)
            else:
                response = await self._call_generic_llama_for_qna(formatted_prompt)
            
            return response.strip()
        except Exception as e:
            print(f"LLaMA QNA 분석 오류: {str(e)}")
            raise

    async def _call_local_model_for_qna(self, prompt: str) -> str:
        try:
            return self.llm_client.local_model.generate_complete(
                prompt=prompt,
                max_tokens=50,
                temperature=0.05,
                top_p=0.8
            )
        except Exception as e:
            print(f"로컬 모델 QNA 분석 실패: {e}")
            raise

    async def _call_bedrock_for_qna(self, prompt: str) -> str:
        try:
            import json
            
            request_body = {
                "prompt": prompt,
                "max_gen_len": 50,
                "temperature": 0.1,
                "top_p": 0.9
            }
            
            response = self.llm_client.bedrock_runtime.invoke_model(
                modelId=self.llm_client.model_id,
                body=json.dumps(request_body),
                contentType="application/json"
            )
            response_body = json.loads(response['body'].read())
            return response_body.get('generation', '')
        except Exception as e:
            print(f"Bedrock QNA 분석 실패: {e}")
            raise

    async def _call_generic_llama_for_qna(self, prompt: str) -> str:
        try:
            if hasattr(self.llm_client, 'generate_complete'):
                if asyncio.iscoroutinefunction(self.llm_client.generate_complete):
                    return await self.llm_client.generate_complete(
                        prompt=prompt,
                        max_tokens=50,
                        temperature=0.1
                    )
                else:
                    return self.llm_client.generate_complete(
                        prompt=prompt,
                        max_tokens=50,
                        temperature=0.1
                    )
            else:
                raise NotImplementedError("지원되지 않는 LLaMA 클라이언트 인터페이스")
        except Exception as e:
            print(f"일반 LLaMA QNA 분석 실패: {e}")
            raise

    async def _analyze_qna_with_generic_llm(self, query: str) -> str:
        if hasattr(self.llm_client, 'generate'):
            return self.llm_client.generate(
                prompt=f"{self.qna_subtype_prompt}\n\nUser: {query}\nAssistant:",
                max_tokens=50,
                temperature=0.1
            )
        else:
            raise NotImplementedError("지원되지 않는 LLM 클라이언트 인터페이스")

# 개발 전용 하네스(테스트 케이스·대화형 모드·__main__)는 제거했다.
# 원 도메인(SecOps) 운영 로그를 픽스처로 담고 있었고, 엔진에서 참조하지 않는 코드였다.
# 엔진이 쓰는 것은 위의 IntentScore · IntentAnalyzer 뿐이다.
