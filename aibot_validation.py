
import os, json, ast, re
import asyncio
import requests
import urllib3
from multiprocessing import Pool
from urllib.parse import quote

try:
    from config_utils import ConfigManager
except ImportError as e:
    print(f"Warning: ConfigManager not available: {e}")

try:
    import pandas as pd
    from tqdm import tqdm
    from openai import OpenAI
except ImportError:
    pass

import sys
import importlib

# 쿼리 검증 플러그인 — 카트리지가 제공 (config [validation] plugin_module/plugin_path).
# 미설정·미설치 시 검증 기능은 비활성화된다.
def _load_validator_plugin():
    try:
        from config_utils import ConfigManager as _CM
        _cfg = _CM().config
        path = _cfg.get('validation', 'plugin_path', fallback='./query-validator/src')
        name = _cfg.get('validation', 'plugin_module', fallback='')
        if not name:
            return None, None
        if path and path not in sys.path:
            sys.path.append(path)
        mod = importlib.import_module(name)
        try:
            svc = importlib.import_module(name + '.core.service')
        except ImportError:
            svc = mod
        return getattr(mod, 'verify_query', None), getattr(svc, 'QueryContext', None)
    except Exception as e:
        print(f"⚠️ 검증 플러그인 로드 실패 ({e}) — 쿼리 검증 비활성화로 동작")
        return None, None

verify_query, QueryContext = _load_validator_plugin()

class Query_validation:
    def __init__(self):
        self.api_initialized = False
        self.api_available = False

    def _initialize_api_config(self):
        if not self.api_initialized:
            try:
                self.config_manager = ConfigManager()
                config = self.config_manager.get_validation_config()
                self.api_key = config['api_key']
                self.base_url = config['base_url']
                self.apps_path = config.get('apps_path', '/api/apps')
                self.session = requests.Session()
                self.session.headers.update({
                    "Authorization": f"Bearer {self.api_key}"
                })
                self.session.verify = False
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                self.api_available = True
            except Exception as e:
                print(f"Warning: API validation not available: {e}")
                self.api_available = False
            finally:
                self.api_initialized = True

    def check_api_server(self):
        self._initialize_api_config()

        if not self.api_available:
            return False

        results = []
        app_name = "experimental"
        url = f"{self.base_url}{self.apps_path}"

        try:
            response = self.session.get(url)
            if response.status_code == 200:
                res = response.json()
                app_list = res['apps']

                if any(item.get("code") == app_name for item in app_list):
                    results = True
                else:
                    print(f"{app_name} is not installed.")
                    results = False
            else:
                print(f"{response.status_code}: {response.text}")
                results = False
        except requests.exceptions.RequestException as e:
            print(str(e))
            results = False

        return results

    def get_installed_apps(self, host: str = "localhost"):
        self._initialize_api_config()

        if not self.api_available:
            return []

        url = f"{self.base_url}{self.apps_path}"

        try:
            response = self.session.get(url)
            if response.status_code == 200:
                res = response.json()
                app_list = res['apps']

                installed_apps = []
                for app in app_list:
                    if app.get('enabled', False):
                        app_entry = f"{app['code']}:{app['version']}"
                        installed_apps.append(app_entry)

                print(f"설치된 앱 목록: {installed_apps}")
                return installed_apps
            else:
                print(f"앱 목록 조회 실패: {response.status_code}: {response.text}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"앱 목록 조회 에러: {str(e)}")
            return []

    def check_specific_app(self, app_code):
        installed_apps = self.get_installed_apps()

        apps_str = str(installed_apps)
        if f'"code": "{app_code}"' in apps_str:
            return True
        return False

    def create_query_context_with_apps(self):
        # QueryContext 는 모듈 상단에서 플러그인 로더가 제공
        context = QueryContext()
        context.apps = self.get_installed_apps()
        return context

    TYPE_KEYWORDS = {"INT", "BOOL", "AGGR_FUNC", "STRING", "FLOAT", "yyyyMMddHHmmss"}
    EXTRA_SYMBOLS = {"...", ",", "=", "[", "]", "{", "}"}
    def is_query_spec_line_by_parser(self, line: str) -> bool:
        line = line.strip()

        if not line or len(line.split()) < 2:
            return False

        bracket_count = line.count("[") + line.count("{")
        type_keyword_found = any(kw in line for kw in self.TYPE_KEYWORDS)
        too_many_special_tokens = sum(sym in line for sym in self.EXTRA_SYMBOLS) >= 5

        return (
            bracket_count >= 2 and
            type_keyword_found and
            too_many_special_tokens
        )


    def is_markdown_table_line(self, line: str) -> bool:
        line = line.strip()

        if line == "..." or not line:
            return True

        pipe_table_pattern = re.compile(r"^\|(?:[^|]+\|)+\s*$")
        plus_dash_pattern = re.compile(r"^\+(?:-+\+)+\s*$")
        dash_header_pattern = re.compile(r"^\|\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?$")
        whitespace_table_pattern = re.compile(r"^\S+\s{2,}\S+(?:\s{2,}\S+)*$")

        return any([
            pipe_table_pattern.match(line),
            plus_dash_pattern.match(line),
            dash_header_pattern.match(line),
            whitespace_table_pattern.match(line),
            self.is_query_spec_line_by_parser(line)
        ])

    def is_query_spec_line_by_parser(self, line: str) -> bool:
        return False

    def split_query_preserving_nested_blocks(self, query: str) -> list[str]:
        """파이프(|) 기준 쿼리 분할. 따옴표/대괄호 안의 파이프는 무시."""
        result = []
        buffer = ""
        bracket_depth = 0
        in_quote = False
        i = 0

        while i < len(query):
            char = query[i]

            if char == '"' and (i == 0 or query[i - 1] != '\\'):
                in_quote = not in_quote
                buffer += char
            elif in_quote:
                buffer += char
            elif char == '[':
                bracket_depth += 1
                buffer += char
            elif char == ']':
                bracket_depth -= 1
                buffer += char
            elif char == '|' and bracket_depth == 0:
                if buffer.strip():
                    result.append(buffer.strip())
                    buffer = ""
            else:
                buffer += char

            i += 1

        if buffer.strip():
            result.append(buffer.strip())

        return result

    def _join_multiline_query(self, query: str) -> str:
        """멀티라인 쿼리를 논리적 단위로 합침. 따옴표/대괄호 컨텍스트 유지."""
        lines = query.splitlines()
        logical_lines = []
        buffer = ""
        in_quote = False
        bracket_depth = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if self.is_markdown_table_line(stripped) and not in_quote and bracket_depth == 0:
                continue

            # 현재 줄을 버퍼에 추가
            if buffer:
                buffer += " " + stripped
            else:
                buffer = stripped

            # 이 줄의 따옴표/대괄호 상태 업데이트
            for ch in stripped:
                if ch == '"':
                    in_quote = not in_quote
                elif not in_quote:
                    if ch == '[':
                        bracket_depth += 1
                    elif ch == ']':
                        bracket_depth -= 1

            # 따옴표와 대괄호가 모두 닫혔으면 논리적 쿼리 완성
            if not in_quote and bracket_depth <= 0:
                logical_lines.append(buffer)
                buffer = ""
                bracket_depth = 0

        # 미완성 버퍼 처리
        if buffer.strip():
            logical_lines.append(buffer)

        return "\n".join(logical_lines)

    def validate_query(self, query: str) -> list:
        # 플러그인 미설정·미설치 = 검증 비활성 — 오류가 아니라 "결과 없음"으로 통과시킨다.
        # (None 상태로 verify_query()를 호출하면 전 청크가 오류 판정되어 수정 루프가 공회전)
        if verify_query is None or QueryContext is None:
            return []
        joined_query = self._join_multiline_query(query)
        query_chunks = []
        for line in joined_query.splitlines():
            if not line.strip():
                continue
            split_chunks = self.split_query_preserving_nested_blocks(line)
            query_chunks.extend(split_chunks)


        results = []
        for chunk in query_chunks:
            try:
                validation_results = verify_query(chunk, QueryContext())

                for result in validation_results:
                    if 'error' in result:
                        results.append({
                            "error": result['error'],
                            "query": result.get('command', chunk),
                            "offset": result.get('offset', 0)
                        })
                    else:
                        results.append({
                            "query": result.get('command', chunk),
                            "is_driver": result.get('is_driver', False),
                            "is_streamable": result.get('is_streamable', False),
                            "query_id": result.get('query_id', 0),
                            "offset": result.get('offset', 0)
                        })

            except Exception as e:
                results.append({
                    "error": str(e),
                    "query": chunk
                })

        return results

    def extract_queries(self, response_text):
        queries = []

        json_objects = re.findall(r'\{.*?\}', response_text, re.DOTALL)
        for obj_str in json_objects:
            try:
                data = json.loads(obj_str)
                if isinstance(data, dict) and "query" in data:
                    queries.append(data["query"])
            except json.JSONDecodeError:
                pass

        if queries:
            return queries

        query_block_pattern = r'```query\s*\n(.*?)\n```'
        matches = re.findall(query_block_pattern, response_text, re.DOTALL)
        for match in matches:
            query = match.strip()
            if query:
                queries.append(query)

        if queries:
            return queries

        alt_patterns = [
            r'``query\s*\n(.*?)\n``',
            r'`query\s*\n(.*?)\n`',
            r'`{4,}query\s*\n(.*?)\n`{4,}'
        ]
        for pattern in alt_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                query = match.strip()
                if query and query not in queries:
                    queries.append(query)

        return queries if queries else []


def extract_error_components(error_info):
    import re
    import json

    error_msg = error_info.get('error', '')

    result = {
        'raw_error': error_msg,
        'type': None,
        'offset': None,
        'note': None,
        'params': None,
        'query': error_info.get('query', ''),
        'command': error_info.get('command', '')
    }

    type_match = re.search(r'type=(\d+)', error_msg)
    if type_match:
        result['type'] = type_match.group(1)

    offset_match = re.search(r'offset=([^,]+)', error_msg)
    if offset_match:
        result['offset'] = offset_match.group(1)

    note_match = re.search(r'note=([^,]+(?:,[^,]*)*?)(?=,\s*params=|$)', error_msg)
    if note_match:
        result['note'] = note_match.group(1).strip()

    params_match = re.search(r'params=(\{[^}]*\})', error_msg)
    if params_match:
        try:
            params_str = params_match.group(1)
            # 오류 메시지는 신뢰 경계 밖 문자열 — 리터럴만 허용 (security-review.md S-5)
            result['params'] = ast.literal_eval(params_str)
        except:
            result['params'] = params_match.group(1)

    return result

def extract_error_guide(error_info):
    components = extract_error_components(error_info)
    return components.get('note', components.get('raw_error', ''))

def format_error_json(error_info):
    import json

    components = extract_error_components(error_info)

    return json.dumps(components, indent=2, ensure_ascii=False)

def translate_error_message(error_info):
    ''' 예시
    {'error': "type=92000, offset=-1, note=paloalto-ngfw-traffic-logs은(는) 지원하지 않는 명령어입니다. 명령어 이름의 철자를 확인하세요., params={'command': 'paloalto-ngfw-traffic-logs'}", 'query': 'paloalto-ngfw-traffic-logs duration=1d from=20251114000000 to=20251114235959', 'offset': 0}
    '''
    print("error_info:", error_info)
    error_type = error_info.get('error', '')
    command = error_info.get('command', '')
    query = error_info.get('query', '')

    original_guide = extract_error_guide(error_info)

    if 'unsupported-command' in error_type or 'type=92000' in error_type:
        match = re.search(r"params=\{[^}]*'command':\s*'([^']+)'\}", error_type)
        cmd = match.group(1) if match else command
        return f"❌ '{cmd}' 명령어는 지원되지 않습니다.\n   📋 가이드: {original_guide}\n   💡 권장: {original_guide}"

    elif 'table-not-found' in error_type:
        table = error_info.get('command', command)
        return f"❌ '{table}' 테이블을 찾을 수 없습니다.\n   📋 가이드: {original_guide}\n   💡 권장: {original_guide}"

    elif 'type=90201' in error_type and 'contains' in query:
        return f"❌ '{query}' 쿼리에서 'contains' 문법 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 'contains \"keyword\"' 대신 '==\"*keyword*\"' 형식을 사용하세요. 예) search name==\"*firewall*\""

    elif 'type=90201' in error_type:
        if 'search' in query and 'not ' in query:
            not_func_pattern = re.findall(r'\bnot\s+(\w+\([^)]+\))', query)
            if not_func_pattern:
                wrong_patterns = []
                correct_patterns = []
                for func_call in not_func_pattern:
                    wrong = f"not {func_call}"
                    correct = f"not({func_call})"
                    wrong_patterns.append(wrong)
                    correct_patterns.append(correct)

                corrected_query = query
                for wrong, correct in zip(wrong_patterns, correct_patterns):
                    corrected_query = corrected_query.replace(wrong, correct)

                guide = "'not' 뒤의 함수 호출에는 괄호가 필요합니다. "
                for wrong, correct in zip(wrong_patterns, correct_patterns):
                    guide += f"'{wrong}' → '{correct}' "
                guide += f"예시: {corrected_query}"

                return f"❌ '{query}' 쿼리에서 파싱 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: {guide}"

        match = re.search(r"params=\{[^}]*'value':\s*'([^']+)'\s*\}", error_type)
        if match:
            value = match.group(1)
            if value.endswith('\n'):
                return f"❌ '{query}' 쿼리에서 파싱 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 쿼리 끝에 불필요한 개행문자가 있습니다. '{value.rstrip()}'"
            else:
                return f"❌ '{query}' 쿼리에서 파싱 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: {original_guide if original_guide else '쿼리 구문을 확인하세요.'}"
        else:
            return f"❌ '{query}' 쿼리에서 파싱 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: {original_guide if original_guide else '쿼리 구문을 확인하세요.'}"

    elif 'type=90202' in error_type:
        return f"❌ '{query}' 쿼리에서 괄호 불일치 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 여는 괄호와 닫는 괄호의 개수를 확인하세요."

    elif 'type=90001' in error_type:
        # if limit in query (example limit==N limit=N .. )-> Edit | limit N
        # sort id limit=2 와 같이 앞에 쿼리는 살려서 sort id | limit 2 이렇게 변경되어야함.
        if 'limit' in query:
            limit_pattern = re.search(r'limit\s*=?\s*(\d+)', query)
            if limit_pattern:
                limit_num = limit_pattern.group(1)
                query_without_limit = re.sub(r'limit\s*=?\s*\d+', '', query).strip()
                corrected_query = f"{query_without_limit} | limit {limit_num}"
                return f"❌ '{query}' 쿼리에서 limit 사용법 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 'limit {limit_num}'를 파이프(|)로 구분하여 사용하세요. 예) {corrected_query}" 
        
        # field="value" 패턴 처리 (= 대신 == 사용해야 함) - 모든 명령어에 적용
        if '=' in query and '==' not in query:
            # field="value" 형태를 field=="value"로 변경
            corrected_query = re.sub(r'(\w+)=("|\')([^"\']*)\2', r'\1==\2\3\2', query)
            if corrected_query != query:  # 변경사항이 있는 경우
                return f"❌ '{query}' 쿼리에서 연산자 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 필드 비교에는 '=' 대신 '==' 연산자를 사용하세요. 예) {corrected_query}"

        # 기존 처리 로직
        match = re.search(r"params=\{[^}]*'command':\s*'([^']+)'\}", error_type)
        cmd = match.group(1) if match else command
        return f"❌ '{query}' 연산자 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: '=' 대신 '==' 연산자를 사용하세요. 예) {cmd}=="

    elif 'type=21603' in error_type and 'sort' in query:
        # sort field desc/asc 패턴 처리
        if re.search(r'sort\s+\w+\s+(desc|asc)', query, re.IGNORECASE):
            match = re.search(r'sort\s+(\w+)\s+(desc|asc)', query, re.IGNORECASE)
            if match:
                field = match.group(1)
                order = match.group(2).lower()
                if order == 'desc':
                    return f"❌ '{query}' 쿼리에서 sort 사용법 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 'sort {field} desc'를 'sort {field}'로 수정하세요. (desc는 기본값이므로 생략)"
                else:  # asc
                    return f"❌ '{query}' 쿼리에서 sort 사용법 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 'sort {field} asc'를 'sort -{field}'로 수정하세요. (오름차순은 필드명 앞에 - 사용)"

        # sort field limit 10 패턴 처리 (field는 -count, count, +field 등 모든 형태)
        if re.search(r'sort\s+[-+]?\w+\s+limit\s+\d+', query):
            match = re.search(r'sort\s+([-+]?\w+)\s+limit\s+(\d+)', query)
            if match:
                sort_field = match.group(1)
                limit_num = match.group(2)
                return f"❌ '{query}' 쿼리에서 sort와 limit 사용법 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 'sort {sort_field} limit {limit_num}'를 'sort {sort_field} | limit {limit_num}'로 수정하세요. 파이프(|)로 구분해야 합니다."

        # 기존 sort limit 패턴 처리
        match = re.search(r'sort\s+limit\s+(\d+)\s+(\S+)', query)
        if match:
            limit_num = match.group(1)
            field = match.group(2)
            return f"❌ '{query}' 쿼리에서 sort limit 사용법 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 'sort limit {limit_num} {field}'를 'sort limit={limit_num} {field}'로 수정하세요."

    elif 'type=10609' in error_type:
        # 날짜 형식 오류 (from/to 옵션의 날짜값이 yyyyMMddHHmmss 형식에 맞지 않음)
        from datetime import datetime as _dt
        param_match = re.search(r"params=\{[^}]*'option':\s*'([^']+)'[^}]*'value':\s*'([^']+)'[^}]*\}", error_type)
        if param_match:
            option_name = param_match.group(1)
            wrong_value = param_match.group(2)
            digits = re.sub(r'\D', '', wrong_value)
            # 8자리 미만이면 자동 보정: from=오늘 00:00:00, to=오늘 23:59:59
            if len(digits) < 8:
                today = _dt.now().strftime("%Y%m%d")
                if option_name == "from":
                    corrected_value = today + "000000"
                else:
                    corrected_value = today + "235959"
            else:
                corrected_value = digits[:14] if len(digits) >= 14 else digits[:12] if len(digits) >= 12 else digits[:10] if len(digits) >= 10 else digits[:8]
            corrected_query = re.sub(rf'{option_name}\s*=\s*{re.escape(wrong_value)}', f'{option_name}={corrected_value}', query)
            return f"❌ '{query}' 쿼리에서 날짜 형식 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: '{option_name}={wrong_value}'를 '{option_name}={corrected_value}'로 수정하세요. (yyyyMMddHHmmss 형식, 14자리) 예) {corrected_query}"
        return f"❌ '{query}' 쿼리에서 날짜 형식 오류가 발생했습니다.\n   📋 가이드: {original_guide}\n   💡 권장: 날짜는 yyyyMMddHHmmss (예: 20240101153045) 형식으로 입력하세요."

    return f"❌ 명령어 '{command}'에 오류가 있습니다.\n   📋 가이드: {original_guide if original_guide else '명령어 구문을 확인하세요.'}\n   💡 권장: 올바른 명령어 형식을 사용하세요."


def validate_and_show_notes(query):
    validation = Query_validation()

    print(f"🔍 쿼리: {query}")
    print("-" * 60)

    result = validation.validate_query(query)

    for item in result:
        if 'error' in item:
            components = extract_error_components(item)

            print(f"❌ 에러 타입: {components['type']}")
            print(f"📋 NOTE 가이드: {components['note']}")
            print(f"🔧 파라미터: {components['params']}")
            print()

            translated = translate_error_message(item)
            print("💡 LLM용 가이드:")
            print(translated)

        else:
            print(f"✅ 성공: {item['query']}")

    print("=" * 60)

def quick_validate(query):
    validation = Query_validation()
    result = validation.validate_query(query)

    for item in result:
        if 'error' in item:
            components = extract_error_components(item)
            print(f"❌ {query}")
            print(f"   Note: {components['note']}")
            return False
        else:
            print(f"✅ {query}")
            return True

if __name__ == "__main__":
    validation = Query_validation()
    print("API 서버 상태:", validation.check_api_server())