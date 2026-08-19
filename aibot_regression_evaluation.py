import os,re
import json
import ast
import asyncio
import pandas as pd
import yaml 
from typing import List, Dict, Tuple, Optional, AsyncGenerator
from tqdm import tqdm
from openai import OpenAI
from config_utils import ConfigManager
from multiprocessing import Pool

from aibot_validation import Query_validation

class EvaluationProcessor:
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = config_path or f"{self.base_dir}/config.ini"
        
        self.client: Optional[OpenAI] = None
        self.available = False
        self.model = None
        self.system_prompt = None
        self.openai_api_key = None
        
        self._initialize_openai()

    def _initialize_openai(self) -> None:
        try:
            self.config_manager = ConfigManager(self.config_path)
            openai_config = self.config_manager.get_openai_config()
            self.openai_api_key = openai_config.get('api_key') or os.getenv("OPENAI_API_KEY")
            
            self.model = self.config_manager.get_model_config().get('model_name', 'gpt-5-mini')
            
            self.system_prompt = self._load_prompt_from_file(f"{self.base_dir}/prompts/regression_g_eval.yaml")
            
            if self.openai_api_key:
                self.client = OpenAI(api_key=self.openai_api_key)
                self.available = True
                print(f"OpenAI 모델 초기화 완료: {self.model}")
            else:
                print("경고: OpenAI API 키가 설정되지 않았습니다.")
                self.available = False
                
        except Exception as e:
            print(f"OpenAI 클라이언트 초기화 오류: {str(e)}")
            self.client = None
            self.available = False

    def _load_prompt_from_file(self, prompt_path: str) -> str:
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                if prompt_path.endswith('.yaml') or prompt_path.endswith('.yml'):
                    yaml_data = yaml.safe_load(f)
                    prompt = self._convert_yaml_to_prompt(yaml_data)
                else:
                    prompt = f.read()
                    
            print(f"프롬프트를 '{prompt_path}'에서 로드했습니다.")
            return prompt
        except Exception as e:
            print(f"프롬프트 파일 로드 중 오류 발생: {str(e)}")
            print("기본 프롬프트를 사용합니다.")
            return self._get_default_prompt()

    def _convert_yaml_to_prompt(self, yaml_data: Dict) -> str:
        prompt_parts = []
        
        if 'system_prompt' in yaml_data:
            prompt_parts.append(yaml_data['system_prompt'].strip())
        
        if 'evaluation_criteria' in yaml_data:
            prompt_parts.append("\n=== EVALUATION CRITERIA ===\n")
            
            for criterion, details in yaml_data['evaluation_criteria'].items():
                prompt_parts.append(f"## {criterion.upper()}")
                prompt_parts.append(f"Description: {details['description']}")
                prompt_parts.append(f"\nInstructions:\n{details['instruction']}")
                
                if 'scoring_scale' in details:
                    prompt_parts.append("\nScoring Scale:")
                    for score, desc in details['scoring_scale'].items():
                        prompt_parts.append(f"  {score}: {desc}")
                
                if 'evaluation_focus' in details:
                    prompt_parts.append("\nEvaluation Focus:")
                    for focus in details['evaluation_focus']:
                        prompt_parts.append(f"  - {focus}")
                
                prompt_parts.append("")
        
        if 'response_format' in yaml_data:
            prompt_parts.append("\n=== RESPONSE FORMAT ===\n")
            prompt_parts.append(f"Type: {yaml_data['response_format']['type']}")
            prompt_parts.append(f"\nStructure:\n{yaml_data['response_format']['structure']}")
        
        if 'evaluation_guidelines' in yaml_data:
            prompt_parts.append("\n=== EVALUATION GUIDELINES ===\n")
            for key, value in yaml_data['evaluation_guidelines'].items():
                prompt_parts.append(f"- {key.title()}: {value}")
        
        if 'quality_assurance' in yaml_data:
            prompt_parts.append("\n=== QUALITY ASSURANCE ===\n")
            for item in yaml_data['quality_assurance']:
                prompt_parts.append(f"- {item}")
        
        if 'example_evaluation' in yaml_data:
            prompt_parts.append("\n=== EXAMPLE EVALUATION ===\n")
            example = yaml_data['example_evaluation']
            prompt_parts.append(f"User Question: {example.get('user_question', '')}")
            prompt_parts.append(f"Model Response: {example.get('model_response', '')}")
            prompt_parts.append(f"Context: {example.get('context', '')}")
            if 'sample_output' in example:
                prompt_parts.append(f"\nSample Output:\n{example['sample_output']}")
        
        return "\n".join(prompt_parts)

    def _get_default_prompt(self) -> str:
        print("경고: 기본 프롬프트가 호출되었습니다. 프롬프트 파일을 확인하세요.")
        return ""

    def load_context(self, context_list: List[str]) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        base_path = f"{self.base_dir}/docs/aibot/yaml"
        context = ""
        document_contents = {}
        document_paths = {}
        subdirs = ['action', 'plan', 'qna']

        for file_name in context_list:
            if not file_name.strip():
                continue

            if not file_name.endswith(".yaml"):
                file_name_with_ext = f"{file_name}.yaml"
            else:
                file_name_with_ext = file_name

            file_found = False

            for subdir in subdirs:
                search_path = os.path.join(base_path, subdir)
                if not os.path.exists(search_path):
                    continue

                for root, dirs, files in os.walk(search_path):
                    for filename in files:
                        if not filename.endswith('.yaml'):
                            continue

                        if (os.path.basename(file_name_with_ext).lower() == filename.lower() or
                            os.path.basename(file_name).lower() in filename.lower()):
                            
                            full_path = os.path.join(root, filename)
                            relative_path = os.path.relpath(full_path, base_path)

                            try:
                                with open(full_path, 'r', encoding='utf-8') as file:
                                    content = file.read()
                                    context += content + "\n"
                                    document_contents[file_name] = content
                                    document_paths[file_name] = relative_path
                                    print(f"✓ {file_name} → {relative_path}")
                                    file_found = True
                                    break
                            except Exception as e:
                                print(f"✗ {file_name}: 읽기 실패 ({relative_path})")

                    if file_found:
                        break
                if file_found:
                    break

            if not file_found:
                print(f"✗ {file_name}: 파일 없음")
                document_contents[file_name] = ""
                document_paths[file_name] = ""

        return context, document_contents, document_paths

    def clean_json_response(self, response_text: str) -> str:
        if "```json" in response_text:
            start_idx = response_text.find("```json") + 7
            end_idx = response_text.find("```", start_idx)
            if end_idx != -1:
                json_content = response_text[start_idx:end_idx].strip()
                return json_content
        elif "```" in response_text:
            start_idx = response_text.find("```") + 3
            end_idx = response_text.find("```", start_idx)
            if end_idx != -1:
                json_content = response_text[start_idx:end_idx].strip()
                return json_content
        
        return response_text.strip()

    async def generate_evaluation_stream(
            self, 
            query: str, 
            answer: str, 
            context_list: List[str]
        ) -> AsyncGenerator[str, None]:
        if not self.available:
            yield "OpenAI API가 구성되지 않았습니다."
            return
        
        context, document_contents = self.load_context(context_list)
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        user_prompt = f"""## User Question: {query}

                    ## Model Response: {answer}

                    ## Expected Reference Documents:
                    Primary (Most Important): {context_list[0] if context_list else 'None'}
                    Additional References: {', '.join(context_list[1:]) if len(context_list) > 1 else 'None'}

                    ## Available Reference Documents Content:"""
        
        for i, (doc_name, doc_content) in enumerate(document_contents.items(), 1):
            priority_label = "PRIMARY" if i == 1 else f"SECONDARY-{i-1}"
            if doc_content:
                user_prompt += f"\n\n### [{priority_label}] {doc_name}:\n{doc_content[:2000]}{'...(truncated)' if len(doc_content) > 2000 else ''}"
            else:
                user_prompt += f"\n\n### [{priority_label}] {doc_name}:\n[DOCUMENT NOT FOUND]"
        
        user_prompt += f"""

                ## Special Evaluation Instructions:
                1. **Reference Quality**: Pay special attention to how well the response utilizes the provided reference documents
                2. **Primary Document Focus**: The first document ({context_list[0] if context_list else 'N/A'}) should be the most relevant - evaluate if it's properly used
                3. **Information Extraction**: Check if key information from reference documents is accurately extracted and presented
                4. **Document Alignment**: Verify that the response aligns with the factual information in the provided documents
                5. **Completeness**: Assess whether the response misses important information that was available in the reference documents

                Please evaluate all criteria (Accuracy, Relevance, Fluency, Conciseness, Reference_Quality) with these considerations in mind."""
                        
        messages.append({"role": "user", "content": user_prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,           
                messages=messages,          
                temperature=0.0,            
                stream=False                
            )

            full_response = response.choices[0].message.content
            yield full_response
                    
        except Exception as e:
            error_message = f"OpenAI 응답 생성 중 오류: {str(e)}"
            print(error_message)
            yield error_message
    
    def calculate_weighted_scores(self, evaluation_results: str) -> Dict[str, float]:
        try:
            clean_json = self.clean_json_response(evaluation_results)
            
            result = json.loads(clean_json)
            weighted_scores = {}

            expected_criteria = ['Accuracy', 'Relevance', 'Fluency', 'Conciseness', 'Reference_Quality']
            
            for category in expected_criteria:
                if category in result:
                    data = result[category]
                    if isinstance(data, dict):
                        if "probability_distribution" in data:
                            prob_dist = data["probability_distribution"]
                            weighted_score = sum(
                                int(score) * (prob / 100) 
                                for score, prob in prob_dist.items()
                            )
                            weighted_scores[category] = round(weighted_score, 2)
                        elif "score" in data:
                            weighted_scores[category] = float(data["score"])
                    elif isinstance(data, (int, float)):
                        weighted_scores[category] = float(data)
            
            return weighted_scores
        except (json.JSONDecodeError, KeyError) as e:
            print(f"점수 계산 중 오류: {str(e)}")
            print(f"정리된 응답: {clean_json[:200]}...")
            return {}

    async def evaluate_dataset_summary(
            self, 
            df: pd.DataFrame
        ) -> Dict[str, float]:
        total_questions = len(df)
        
        intent_col = None
        response_time_col = None
        
        for col in df.columns:
            col_clean = col.strip()
            col_lower = col_clean.lower()
            
            if col_clean == '의도타입' or col_lower == 'intent_type':
                intent_col = col_clean
            elif col_clean == '응답시간(초)' or col_lower in ['response_time', 'time_taken']:
                response_time_col = col_clean
        
        intent_distribution = {}
        if intent_col and intent_col in df.columns:
            intent_distribution = df[intent_col].value_counts().to_dict()
        else:
            print(f"경고: 의도타입 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {list(df.columns)}")
            intent_distribution = {"unknown": total_questions}
        
        avg_response_time = 0
        if response_time_col and response_time_col in df.columns:
            time_data = df[response_time_col].dropna()
            if len(time_data) > 0:
                avg_response_time = time_data.mean()
        
        print(f"데이터셋 요약:")
        print(f"  총 질문 수: {total_questions}")
        print(f"  의도 분포: {intent_distribution}")
        print(f"  평균 응답 시간: {avg_response_time:.2f}초")
        
        sample_pairs = []
        
        if intent_col and intent_col in df.columns:
            for intent in df[intent_col].unique():
                if pd.isna(intent):
                    continue
                intent_data = df[df[intent_col] == intent]
                if len(intent_data) > 0:
                    samples = intent_data.head(2)
                    for _, row in samples.iterrows():
                        sample_pairs.append({
                            'intent': str(intent),
                            'question': row.get('질문', ''),
                            'answer': row.get('답변', ''),
                            'reference_docs': row.get('참조문서', '')
                        })
        else:
            samples = df.head(4)
            for _, row in samples.iterrows():
                sample_pairs.append({
                    'intent': 'general',
                    'question': row.get('질문', ''),
                    'answer': row.get('답변', ''),
                    'reference_docs': row.get('참조문서', '')
                })
        
        print(f"샘플 수: {len(sample_pairs)}개")
        
        user_prompt = f"""
                ## Dataset Summary for Evaluation

                **Dataset Overview:**
                - Total Questions: {total_questions}
                - Intent Distribution: {intent_distribution}
                - Average Response Time: {avg_response_time:.2f} seconds

                **Sample Question-Answer Pairs:**
                """
                    
        for i, pair in enumerate(sample_pairs, 1):
            user_prompt += f"""
                ### Sample {i} ({pair['intent'].upper()})
                **Question:** {pair['question']}
                **Answer:** {pair['answer'][:500]}{'...' if len(pair['answer']) > 500 else ''}
                **Reference Docs:** {pair['reference_docs']}

                """

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            print("OpenAI API 호출 중...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                stream=False
            )
            
            full_response = response.choices[0].message.content
            print(f"API 응답 수신: {len(full_response)}자")
            print(f"응답 미리보기: {full_response[:200]}...")
            
            scores = self.calculate_weighted_scores(full_response)
            print(f"파싱된 점수: {scores}")
            
            mapped_scores = {}
            
            score_mapping = {
                'Overall_Accuracy': '전체_정확성',
                'Overall_Relevance': '전체_관련성',
                'Overall_Consistency': '전체_일관성',
                'Reference_Utilization': '참조문서_활용도',
                'Response_Completeness': '응답_완성도',
                'Accuracy': '전체_정확성',
                'Relevance': '전체_관련성',
                'Consistency': '전체_일관성',
                'Reference_Quality': '참조문서_활용도',
                'Completeness': '응답_완성도'
            }
            
            for eng_key, kor_key in score_mapping.items():
                if eng_key in scores:
                    mapped_scores[kor_key] = scores[eng_key]
            
            for key, value in scores.items():
                if key not in score_mapping:
                    mapped_scores[key] = value
            
            if len(mapped_scores) > 0:
                non_zero_scores = [v for v in mapped_scores.values() if v > 0]
                if non_zero_scores:
                    mapped_scores['전체_평균'] = round(sum(non_zero_scores) / len(non_zero_scores), 2)
            
            print(f"최종 매핑된 점수: {mapped_scores}")
            return mapped_scores
            
        except Exception as e:
            print(f"데이터셋 종합 평가 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

    async def evaluate_single_row(
            self, 
            query: str, 
            answer: str, 
            context_list: List[str],
            intent_type: str = None
        ) -> Dict[str, float]:
        is_plan_step = '[Plan Step' in query
        
        if is_plan_step:
            import re
            step_match = re.search(r'\[Plan Step \d+\] (.+)', query)
            if step_match:
                actual_query = step_match.group(1)
                print(f"Plan 단계에서 실제 쿼리 추출: {actual_query}")
            else:
                actual_query = query
        else:
            actual_query = query
        
        if isinstance(context_list, list) and context_list:
            context_list = [context_list[0]]
        else:
            context_list = []

        context, document_contents, document_paths = self.load_context(context_list)

        loaded_count = len([v for v in document_contents.values() if v])
        intent_info = f"[{intent_type.upper()}]" if intent_type else "[UNKNOWN]"
        step_info = "[PLAN-STEP]" if is_plan_step else ""
        print(f"{intent_info}{step_info} 참조문서: {loaded_count}/{len(context_list)}개 로드됨")

        messages = [{"role": "system", "content": self.system_prompt}]

        if is_plan_step:
            user_prompt = f"""## User Question (Plan Step): {actual_query}
                        ## Original Plan Context: This is a step from a larger plan execution.
                        ## Model Response: {answer}
                        """
        else:
            user_prompt = f"""## User Question: {actual_query}
                        ## Model Response: {answer}
                        """

        if context_list:
            user_prompt += f"\n\n## Reference Documents (Ground Truth):"
            for i, (doc_name, doc_content) in enumerate(document_contents.items(), 1):
                if doc_content:
                    user_prompt += f"\n\n### [PRIMARY] {doc_name}:\n{doc_content[:2000]}{'...(truncated)' if len(doc_content) > 2000 else ''}"
                else:
                    user_prompt += f"\n\n### [PRIMARY] {doc_name}:\n[DOCUMENT NOT FOUND]"

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,           
                messages=messages,          
                temperature=0.0,            
                stream=False                
            )

            full_response = response.choices[0].message.content
            scores = self.calculate_weighted_scores(full_response)

            if scores:
                mapped_scores = {}
                score_mapping = {
                    'Accuracy': '정확성',
                    'Relevance': '관련성', 
                    'Fluency': '유창성',
                    'Conciseness': '간결성'
                }
                for eng_key, kor_key in score_mapping.items():
                    if eng_key in scores:
                        mapped_scores[kor_key] = scores[eng_key]

                return mapped_scores

            return {}
                        
        except Exception as e:
            print(f"평가 중 오류: {str(e)}")
            return {}

    async def evaluate_plan_comprehensive(
            self, 
            original_question: str,
            comprehensive_context: str,
            context_list: List[str],
            step_count: int
        ) -> Dict[str, float]:
        context, document_contents, document_paths = self.load_context(context_list)

        loaded_count = len([v for v in document_contents.values() if v])
        print(f"[PLAN-COMPREHENSIVE] 참조문서: {loaded_count}/{len(context_list)}개 로드됨")

        messages = [{"role": "system", "content": self.system_prompt}]

        user_prompt = f"""## Plan Comprehensive Evaluation Request

                        {comprehensive_context}

                        ## Reference Documents (if available):"""

        if context_list:
            for i, (doc_name, doc_content) in enumerate(document_contents.items(), 1):
                if doc_content:
                    user_prompt += f"\n\n### [REFERENCE] {doc_name}:\n{doc_content[:2000]}{'...(truncated)' if len(doc_content) > 2000 else ''}"
                else:
                    user_prompt += f"\n\n### [REFERENCE] {doc_name}:\n[DOCUMENT NOT FOUND]"

        user_prompt += f"""

                    ## Evaluation Instructions:
                    Please evaluate this Plan holistically, considering:
                    1. **Plan Quality**: How well does the generated plan address the original question?
                    2. **Step Execution**: How well were the individual steps executed?
                    3. **Coherence**: Do the step results work together to answer the original question?
                    4. **Completeness**: Does the overall result fully address what was asked?
                    5. **Reference Utilization**: How well were reference documents used throughout the process?

                    Focus on the END-TO-END effectiveness of the Plan rather than individual step quality."""

        messages.append({"role": "user", "content": user_prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,           
                messages=messages,          
                temperature=0.0,            
                stream=False                
            )

            full_response = response.choices[0].message.content
            scores = self.calculate_weighted_scores(full_response)

            if scores:
                mapped_scores = {}
                score_mapping = {
                    'Accuracy': '정확성',
                    'Relevance': '관련성', 
                    'Fluency': '유창성',
                    'Conciseness': '간결성'
                }
                for eng_key, kor_key in score_mapping.items():
                    if eng_key in scores:
                        mapped_scores[kor_key] = scores[eng_key]

                return mapped_scores

            return {}
                        
        except Exception as e:
            print(f"Plan 종합 평가 중 오류: {str(e)}")
            return {}

class DataFrameEvaluator:
    
    def __init__(self, evaluation_processor: EvaluationProcessor):
        self.eval_processor = evaluation_processor

    async def process_single_file(self, file_path: str, processes: int = 1) -> None:
        try:
            df = pd.read_csv(file_path)
            
            print(f"파일 로드 완료: {len(df)}개 행")
            print(f"컬럼: {list(df.columns)}")
            
            plan_step_mask = df['질문'].str.contains(r'\[Plan Step \d+\]', regex=True, na=False)
            original_questions = df[~plan_step_mask]
            plan_steps = df[plan_step_mask]
            
            print(f"원본 질문: {len(original_questions)}개")
            print(f"Plan 단계: {len(plan_steps)}개")
            
            existing_eval_columns = [col for col in df.columns if col.startswith('전체_')]
            
            if existing_eval_columns:
                df = df.drop(columns=existing_eval_columns)
            
            print("질문-답변 쌍에 대한 개별 평가 시작...")
            
            all_scores = []
            
            for idx, row in df.iterrows():
                is_plan_step = '[Plan Step' in str(row.get('질문', ''))
                question = row.get('질문', '')
                answer = row.get('답변', '')
                intent_type = row.get('의도타입', 'unknown')
                
                print(f"{'Plan 단계' if is_plan_step else '원본'} 평가 진행: {idx + 1}/{len(df)} - {question[:50]}...")

                if intent_type.upper() == 'ACTION':
                    try:
                        validator = Query_validation()
                        filtered_blocks = extract_query_blocks(answer)

                        queries = []
                        if filtered_blocks:
                            try:
                                data = json.loads(filtered_blocks)
                                if isinstance(data, list):
                                    queries = [obj.get("query") for obj in data if isinstance(obj, dict) and "query" in obj]
                                elif isinstance(data, dict) and "query" in data:
                                    queries = [data["query"]]
                            except Exception as e:
                                print(f"JSON decode 실패: {e}")

                        if queries:
                            results = []
                            for q in queries:
                                results.extend(validator.validate_query(q))

                            if all("error" not in r for r in results):
                                df.at[idx, '코드_검증'] = "PASS"
                            else:
                                df.at[idx, '코드_검증'] = "FAIL"
                        else:
                            df.at[idx, '코드_검증'] = "NO_QUERY"

                    except Exception as e:
                        print(f"코드 검증 오류: {e}")
                        df.at[idx, '코드_검증'] = "ERROR"
                
                reference_docs = row.get('실제참조문서', '')
                if isinstance(reference_docs, str) and reference_docs.strip():
                    context_list = [doc.strip() for doc in reference_docs.split(',') if doc.strip()]
                else:
                    context_list = []

                if context_list:
                    context_list = [context_list[0]]
                
                try:
                    scores = await self.eval_processor.evaluate_single_row(
                        query=question,
                        answer=answer, 
                        context_list=context_list,
                        intent_type=intent_type
                    )
                    
                    if scores:
                        all_scores.append(scores)
                        print(f"  평가 완료: {scores}")
                        
                except Exception as e:
                    print(f"  평가 중 오류: {e}")
                    continue
            
            if all_scores:
                print(f"개별 평가 완료: {len(all_scores)}개 성공")
                
                final_scores = {}
                all_keys = set()
                for score_dict in all_scores:
                    all_keys.update(score_dict.keys())
                
                basic_criteria = ['정확성', '관련성', '유창성', '간결성']
                
                for key in all_keys:
                    if key in basic_criteria:
                        values = [score_dict[key] for score_dict in all_scores if key in score_dict and score_dict[key] > 0]
                        if values:
                            final_scores[key] = round(sum(values) / len(values), 2)
                
                print(f"최종 집계 점수: {final_scores}")
                
                for score_name, score_value in final_scores.items():
                    df[f"전체_{score_name}"] = score_value
                
                if final_scores:
                    df['전체_평균'] = round(sum(final_scores.values()) / len(final_scores), 2)
                
                df.to_csv(file_path, index=False, encoding='utf-8')
                print(f"개별 평가 결과 저장 완료: {file_path}")
                
            else:
                print("평가된 결과가 없습니다.")
                    
        except Exception as e:
            print(f"파일 처리 오류 {file_path}: {e}")
            import traceback
            traceback.print_exc()

    def _group_plan_questions(self, df):
        plan_groups = []
        plan_rows = df[df['의도타입'].str.upper() == 'PLAN']
        
        for idx, plan_row in plan_rows.iterrows():
            original_question = plan_row['질문']
            plan_answer = plan_row['답변']
            
            step_rows = []
            next_idx = idx + 1
            
            while next_idx < len(df):
                if next_idx >= len(df):
                    break
                    
                next_row = df.iloc[next_idx]
                if '[Plan Step' in str(next_row.get('질문', '')):
                    step_rows.append({
                        'step_number': len(step_rows) + 1,
                        'question': next_row['질문'],
                        'answer': next_row['답변'],
                        'intent_type': next_row['의도타입'],
                        'sources': next_row.get('실제참조문서', ''),
                        'time_taken': next_row.get('응답시간(초)', 0)
                    })
                    next_idx += 1
                else:
                    break
            
            plan_groups.append({
                'original_question': original_question,
                'plan_answer': plan_answer,
                'step_results': step_rows,
                'reference_docs': plan_row.get('실제참조문서', ''),
                'total_time': plan_row.get('응답시간(초)', 0) + sum(step.get('time_taken', 0) for step in step_rows)
            })
        
        print(f"Plan 그룹 생성 완료: {len(plan_groups)}개 그룹")
        for i, group in enumerate(plan_groups):
            print(f"  그룹 {i+1}: {len(group['step_results'])}개 단계")
        
        return plan_groups

    async def process_folder(self, logs_folder: str, processes: int = 1) -> None:
        for dirpath, _, filenames in os.walk(logs_folder):
            for filename in filenames:
                if not filename.endswith(".csv"):
                    continue
                    
                file_path = os.path.join(dirpath, filename)
                await self.process_single_file(file_path, processes)


def extract_query_blocks(response_text: str):
    match = re.search(r'\[\s*\{.*?\}\s*\]', response_text, re.DOTALL)
    if not match:
        return ""

    block = match.group(0)
    try:
        parsed = json.loads(block)
        return json.dumps(parsed, ensure_ascii=False)
    except Exception as e:
        print(f"[extract_query_blocks] JSON 파싱 실패: {e}")
        return ""

async def test_evaluation():
    eval_processor = EvaluationProcessor()
    
    if not eval_processor.available:
        print("OpenAI API가 설정되지 않았습니다.")
        return
    
    query = "3월 25일 부터 3월 31일까지와 4월 1일 부터 4월 7일까지의 로그 수집 결과를 비교해줘"
    answer = """
    안녕하세요! 로그 수집 결과를 비교해 드릴게요!

    3월 25일부터 3월 31일까지의 로그 수집 결과와 4월 1일부터 4월 7일까지의 로그 수집 결과를 비교해 보겠습니다.

    **비교 기준:**
    * 로그 수집량 (수집된 로그의 총 개수)
    * 로그 수집 속도 (수집된 로그의 평균 시간)
    """
    
    context = [
        'action/query-date-range-log-collection-trend.yaml', 
        'action/api-logger-list-by-log-schema.yaml'
    ]
    
    full_response = ""
    async for response in eval_processor.generate_evaluation_stream(query, answer, context):
        full_response += response
    
    print("평가 결과:")
    print(full_response)
    print("\n계산된 점수:")
    print(eval_processor.calculate_weighted_scores(full_response))

async def run_folder_evaluation(logs_folder: str = "./logs/", processes: int = 1):
    eval_processor = EvaluationProcessor()
    df_evaluator = DataFrameEvaluator(eval_processor)
    await df_evaluator.process_folder(logs_folder, processes)

async def run_file_evaluation(file_path: str, processes: int = 1):
    eval_processor = EvaluationProcessor()
    df_evaluator = DataFrameEvaluator(eval_processor)
    await df_evaluator.process_single_file(file_path, processes)


if __name__ == "__main__":
    asyncio.run(test_evaluation())