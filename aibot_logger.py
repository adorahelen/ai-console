from datetime import datetime
from enum import Enum
import pandas as pd
import os
import stat

class LOGGERMODE(Enum):
    DEFAULT = 0
    LOG_USER_ONLY = 1
    LOG_ALL_ONLY = 2

class ChatLogger:
    def __init__(self, model, mode = LOGGERMODE.DEFAULT):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dirPath = f"{base_dir}/logs"
        self.fileName = "logs"
        self.ext = "csv"
        self.model = model
        self.mode = mode
        
    def changeMode(self, mode: LOGGERMODE):
        self.mode = mode
        return
    
    
    def log(self, user, question, answer, answerTime, context, concepts, related_concepts, connection, is_cache=False):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        dateStr = datetime.now().strftime('%y%m%d')
        savePath = f"{self.dirPath}/{dateStr}"
        
        userName = user
        
        if userName is None:
            userName = 'REST_API'
            connection = 'REST_API'

        data = {
            '날짜': [now],
            '사용자': [userName],
            '활용모델': [self.model],
            '질문': [question],
            '답변': [answer],
            '응답시간(초)': [answerTime],
            '추출개념': [concepts],
            '지식그래프': [related_concepts],
            '참조문서': [context],
            '캐싱': [is_cache],
            '접속방식': [connection],
            '비고': ['']
        }
        
        df = pd.DataFrame(data)
        try:
            os.makedirs(self.dirPath, exist_ok=True)
            os.makedirs(savePath, exist_ok=True, mode=0o777)
            current_mode = os.stat(savePath).st_mode
            current_permissions = stat.S_IMODE(current_mode)
            if current_permissions != 0o777:
                os.chmod(savePath, 0o777)
        except PermissionError as e:
            print(f"⚠️ 권한 설정 실패: {e}")
            
        try:
            if self.mode == LOGGERMODE.DEFAULT or self.mode == LOGGERMODE.LOG_ALL_ONLY:
                file_name = f'{savePath}/{self.fileName}_{dateStr}.{self.ext}'
                
                existing_df = pd.read_csv(file_name)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_csv(file_name, index=False, encoding='utf-8')
                print(f"✅ 전체 응답이 CSV 파일에 저장됨: {file_name}")
                
        except Exception as e:
            print(f"전체 응답 파일이 없습니다. 새 파일을 생성합니다.")
            df.to_csv(file_name, index=False, encoding='utf-8')
            os.chmod(file_name, 0o777)
            
        try:
            if self.mode == LOGGERMODE.DEFAULT or self.mode == LOGGERMODE.LOG_USER_ONLY:
                file_name = f'{savePath}/{self.fileName}_{dateStr}_{userName}.{self.ext}'
                
                existing_df = pd.read_csv(file_name)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
                updated_df.to_csv(file_name, index=False, encoding='utf-8') 
                print(f"✅ 개별 응답이 CSV 파일에 저장됨: {file_name}")
                
        except Exception as e:
            print(f"개별 응답 파일이 없습니다. 새 파일을 생성합니다.")
            df.to_csv(file_name, index=False, encoding='utf-8')
            os.chmod(file_name, 0o777)