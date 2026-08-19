import re
from aibot_validation import Query_validation

def extract_queries(response_text):
    lines = response_text.split('\n')
    queries = []
    current_query = []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('table') or line.startswith('join') or line.startswith('|'):
            if line.startswith('table') or line.startswith('join'):
                if current_query:
                    queries.append('\n'.join(current_query))
                    current_query = []
            
            if '//' in line:
                line = line.split('//')[0].strip()
            
            if line:
                current_query.append(line)
    
    if current_query:
        queries.append('\n'.join(current_query))
    
    return queries

# 테스트
llm_response = """
답변:  Hello there! :로봇_얼굴: You're looking for a way to display the differences in log data by type over the last 15 days, sorted by time!
You can use this query:
table duration=15d *:FW_* | stats count by _time, type
In this query, duration=15d specifies that you want to look at the last 15 days of data. The *:FW_* is a wildcard namespace that refers to all nodes in the cluster and all tables starting with FW_. The stats count by _time, type command will group the results by time and type, and display the count for each group.
This query will give you a table with two columns: _time and type, and one row for each unique combination of time and type. You can sort the results by time using the sort command:
table duration=15d *:FW_* | stats count by _time, type | sort _time
This will give you a table with the same columns as before, but sorted in chronological order.
If you want to display more information about each log entry, such as the source and destination IP addresses, you can add additional fields to the stats command:
table duration=15d *:FW_* | stats count, src_ip, dst_ip by _time, type
This will give you a table with four columns: _time, type, src_ip, and dst_ip.
Let me know if you have any further questions or need more help! :미소짓는_상기된_얼굴:

"""

queries = extract_queries(llm_response)

validation = Query_validation()

for i, query in enumerate(queries, 1):
    result = validation.validate_query(query)
    errors = [item for item in result if 'error' in item]

    for error_item in errors:
        print(f"오류 발생 쿼리: {error_item['command']}")
        print(f"오류 내용: {error_item['error']}")
        print("-" * 50)