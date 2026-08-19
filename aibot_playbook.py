
import json
import csv
import requests
import urllib3
import uuid
import time
from datetime import datetime
from pathlib import Path
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class PlaybookClient:
    def __init__(self, base_url="https://localhost:443"):
        self.base_url = base_url
        self.endpoint = f"{base_url}/api/ai/chats/regression"
        
        self.setup_output_directory()
        
    def setup_output_directory(self):
        dateStr = datetime.now().strftime('%y%m%d')
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.work_dir = os.path.join(base_dir, 'playbooks', dateStr)
        os.makedirs(self.work_dir, exist_ok=True)
        
    def send_playbook_request(self, question: str) -> dict:
        
        user_guid = str(uuid.uuid4())
        
        payload = {
            "user_guid": user_guid,
            "type": "playbook",
            "question": question,
            "llm_model": "gpt-oss",
            "prompt_count": 0,
            "prompt_token": 0,
            "locale": "ko",
            "messages": []
        }
        
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Bearer 1885c878-2264-4714-84e4-27a615ff1020'
        }
        
        params = {
            "group": "test"
        }
        
        try:
            print(f"플레이북 생성 요청 중...")
            print(f"질문: {question[:100]}...")
            
            response = requests.post(
                self.endpoint, 
                json=payload, 
                headers=headers, 
                params=params,
                verify=False,
                timeout=60
            )
            
            print(f"응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"플레이북 생성 완료")
                return result
            else:
                print(f"오류: {response.status_code} - {response.text}")
                return {"error": f"API 호출 실패: {response.status_code}"}
                
        except Exception as e:
            print(f"요청 실패: {str(e)}")
            return {"error": str(e)}
    
    def extract_json_from_answer(self, answer: str) -> dict:
        try:
            start_idx = answer.find('{')
            end_idx = answer.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = answer[start_idx:end_idx]
                return json.loads(json_str)
            else:
                print("답변에서 JSON을 찾을 수 없습니다")
                return None
                
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류: {e}")
            return None
    
    def save_to_csv(self, question: str, answer: str, playbook_json: dict, time_taken: float):
        
        dateStr = datetime.now().strftime('%y%m%d')
        csv_file = os.path.join(self.work_dir, f'playbooks_{dateStr}.csv')
        
        playbook_name = playbook_json.get('name', '') if playbook_json else ''
        step_count = len(playbook_json.get('steps', [])) if playbook_json else 0
        priority = playbook_json.get('priority', '') if playbook_json else ''
        
        row_data = [
            datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            question,
            playbook_name,
            priority,
            step_count,
            time_taken,
            answer,
            json.dumps(playbook_json, ensure_ascii=False, indent=2) if playbook_json else ''
        ]
        
        headers = [
            '생성시간', '질문', '플레이북명', '우선순위', 
            '단계수', '생성시간(초)', '전체답변', 'JSON데이터'
        ]
        
        try:
            file_exists = os.path.exists(csv_file)
            
            with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                if not file_exists:
                    writer.writerow(headers)
                    print(f"새 CSV 파일 생성: {csv_file}")
                
                writer.writerow(row_data)
            
            print(f"결과가 CSV에 저장됨: {csv_file}")
            return csv_file
            
        except Exception as e:
            print(f"CSV 저장 실패: {e}")
            return None
    
    def create_playbook(self, question: str) -> bool:
        
        print(f"=== 플레이북 생성 시작 ===")
        
        start_time = time.time()
        
        result = self.send_playbook_request(question)
        
        if 'error' in result:
            print(f"생성 실패: {result['error']}")
            return False
        
        answer = result.get('answer', '')
        detected_intent = result.get('intent', 'unknown')
        sources = result.get('sources', [])
        
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        
        print(f"응답 완료 (소요시간: {time_taken}초)")
        print(f"감지된 의도: {detected_intent}")
        
        if sources:
            print(f"참조 소스: {len(sources)}개")
        
        playbook_json = self.extract_json_from_answer(answer)
        
        if playbook_json:
            print(f"플레이북 정보:")
            print(f"  이름: {playbook_json.get('name', 'N/A')}")
            print(f"  우선순위: {playbook_json.get('priority', 'N/A')}")
            print(f"  단계수: {len(playbook_json.get('steps', []))}개")
        else:
            print("JSON 추출 실패")
        
        csv_file = self.save_to_csv(question, answer, playbook_json, time_taken)
        
        if csv_file:
            print(f"=== 플레이북 생성 완료 ===")
            print(f"결과 파일: {csv_file}")
            return True
        else:
            print("CSV 저장 실패")
            return False
    
    def batch_create_playbooks(self, questions: list) -> int:
        
        print(f"=== 배치 플레이북 생성 시작 ===")
        print(f"총 {len(questions)}개 질문 처리 예정")
        
        success_count = 0
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] 처리 중...")
            
            success = self.create_playbook(question)
            
            if success:
                success_count += 1
                print(f"✅ 성공")
            else:
                print(f"❌ 실패")
            
            if i < len(questions):
                print("3초 대기...")
                time.sleep(3)
        
        print(f"\n=== 배치 처리 완료 ===")
        print(f"성공: {success_count}/{len(questions)}개")
        
        return success_count

if __name__ == "__main__":
    client = PlaybookClient()
    
    question = """
    보안 상황: 웹 서버에서 SQL 인젝션 공격이 탐지되었습니다. 
    공격자는 IP 주소 192.168.1.100에서 /login 페이지를 통해 악성 SQL 쿼리를 반복적으로 시도하고 있으며, 
    데이터베이스 접근 로그에서 비정상적인 쿼리 패턴이 확인되었습니다. 
    현재 공격이 진행 중이므로 즉시 차단하고 피해 범위를 조사한 후 보안 강화 조치를 수행해야 합니다.
    
    위 상황에 대한 보안 대응 플레이북을 JSON으로 생성하세요.
    """
    
    client.create_playbook(question)
    
    questions = [
        "DDoS 공격 탐지 시 대응 플레이북을 생성해주세요.",
        "랜섬웨어 감염 의심 시 초기 대응 플레이북을 만들어주세요.",
        "내부 계정 무단 사용 탐지 시 대응 플레이북을 생성해주세요."
    ]
    
