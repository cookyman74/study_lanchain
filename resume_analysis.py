import os
import time
import json
import hashlib
import threading
from queue import Queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
import fitz  # PyMuPDF
import csv
from openpyxl import load_workbook, Workbook

from dotenv import load_dotenv
load_dotenv()

# 환경 변수 사용
openai_api_key = os.getenv('OPENAI_API_KEY')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

directory_to_watch = os.getenv('DIRECTORY_TO_WATCH')
excel_file_path = os.getenv('EXCEL_FILE_PATH')
hash_record_path = os.getenv('HASH_RECORD_PATH')
csv_file_path = os.getenv('CSV_FILE_PATH')

# 특정 디렉토리 설정
DIRECTORY_TO_WATCH = directory_to_watch
EXCEL_FILE_PATH = excel_file_path
HASH_RECORD_PATH = hash_record_path
CSV_FILE_PATH = csv_file_path

# 해시 기록을 저장하기 위한 파일 로드
if os.path.exists(HASH_RECORD_PATH):
    with open(HASH_RECORD_PATH, "r") as f:
        processed_files = json.load(f)
else:
    processed_files = {}

# 큐 설정
file_queue = Queue()

class ResumeHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return None
        elif event.src_path.endswith(".pdf"):
            file_queue.put(event.src_path)

def process_files_from_queue():
    while True:
        file_path = file_queue.get()
        if file_path is None:
            break
        process_resume(file_path)
        file_queue.task_done()

def process_resume(file_path):
    file_hash = calculate_file_hash(file_path)

    if file_hash in processed_files:
        print(f"File {file_path} has already been processed.")
        return

    analysis_result = analyze_resume_with_langchain(file_path)
    if analysis_result:
        update_excel_with_result(analysis_result)
        processed_files[file_hash] = file_path
        save_hash_records()
    else:
        print(f"Error processing file {file_path}.")

def calculate_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def save_hash_records():
    with open(HASH_RECORD_PATH, "w") as f:
        json.dump(processed_files, f)

def analyze_resume_with_langchain(file_path):
    resume_text = extract_text_from_pdf(file_path)

    # LangChain 사용하여 텍스트 분석
    prompt = ChatPromptTemplate.from_template(
        """
        이력서 내용: {RESUME_TEXT}
        이력서에서 다음 정보를 추출하고 JSON 형식으로 반환하세요.
        - 지원자 이름
        - 나이
        - 경력
        - 핵심기술력
        - 특징

        JSON 형식 예시:
        {{
            "지원자 이름": "홍길동",
            "나이": "30",
            "경력": "10년",
            "핵심기술력": "Python, Machine Learning",
            "특징": "팀 리더 경험"
        }}
        """
    )

    chain = (
            prompt
            | ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )
            | JsonOutputParser()
    )

    result = chain.invoke({"RESUME_TEXT": resume_text})
    if isinstance(result, dict):
        print("999999: ", result)
        return result
    else:
        print("Error: Unexpected output format.")
        return {}

def extract_text_from_pdf(file_path):
    # PyMuPDF를 사용하여 PDF에서 텍스트 추출
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def update_excel_with_result(result):
    lock = threading.Lock()
    with lock:
        # 엑셀 파일 업데이트
        if os.path.exists(EXCEL_FILE_PATH):
            workbook = load_workbook(EXCEL_FILE_PATH)
        else:
            workbook = Workbook()
            sheet = workbook.active
            sheet.append(["지원자 이름", "나이", "경력", "핵심기술력", "특징"])  # 헤더 추가

        sheet = workbook.active
        # JSON 데이터를 문자열로 변환하여 저장
        name = result.get("지원자 이름", "")
        age = result.get("나이", "")
        experience = result.get("경력", "")
        skills = result.get("핵심기술력", "")
        characteristics = result.get("특징", "")

        sheet.append([name, age, experience, skills, characteristics])
        workbook.save(EXCEL_FILE_PATH)

def update_csv_with_result(result):
    # CSV 파일 업데이트
    file_exists = os.path.exists(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["지원자 이름", "나이", "경력", "핵심기술력", "특징"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

if __name__ == "__main__":
    event_handler = ResumeHandler()
    observer = Observer()
    observer.schedule(event_handler, path=DIRECTORY_TO_WATCH, recursive=False)
    observer.start()

    # 파일 처리 스레드 시작
    processing_thread = threading.Thread(target=process_files_from_queue, daemon=True)
    processing_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        file_queue.put(None)  # 큐를 종료하기 위해 None 추가
        processing_thread.join()
    observer.join()
