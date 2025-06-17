#!/bin/bash

# 로그 파일 경로 설정
LOG_FILE="/workspace/output/system.log"

# 로그 함수 정의
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# 시작 로그 기록
log_message "=== 스크립트 실행 시작 ==="

# 프로젝트 디렉토리로 이동
cd /workspace
log_message "작업 디렉토리로 이동 완료"

# AI_Lambda 디렉토리 생성 및 이동
mkdir -p AI_Lambda
cd AI_Lambda
log_message "AI_Lambda 디렉토리 생성 및 이동 완료"

# AI_Lambda 레포지토리 클론
git clone https://github.com/OptiQuantTeam/AI_Lambda.git .
log_message "AI_Lambda 레포지토리 클론 완료"

# 상위 디렉토리로 이동하여 AI 학습 실행
cd ..
log_message "AI 학습 시작"
python3 src/main2.py
log_message "AI 학습 완료"

# 학습된 모델 파일을 AI_Lambda/model 디렉토리로 복사
cp -r saved_model/* AI_Lambda/model/
log_message "모델 파일 복사 완료"

# 메타데이터 파일을 S3에 업로드
log_message "메타데이터 파일 S3 업로드 시작"
if [ -d "saved_model/metadata" ]; then
    # S3 버킷 이름 설정
    S3_BUCKET="optiquant-ai-metadata"

    # metadata 디렉토리의 모든 json 파일을 S3에 업로드
    for file in saved_model/metadata/*.json; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            # S3에 파일 업로드
            if aws s3 cp "$file" "s3://${S3_BUCKET}/${filename}"; then
                log_message "메타데이터 파일 업로드 성공: ${filename}"
            else
                log_message "메타데이터 파일 업로드 실패: ${filename}"
            fi
        fi
    done
    log_message "메타데이터 파일 S3 업로드 완료"
else
    log_message "메타데이터 디렉토리를 찾을 수 없습니다"
fi

# AI_Lambda 디렉토리로 이동
cd AI_Lambda

# 변경사항 커밋 및 푸시
if git add . && git commit -m "Update model files" && git push; then
    log_message "모델 파일이 성공적으로 업로드되었습니다."
    # 컨테이너 종료
    log_message "컨테이너 종료"
    exit 0
else
    log_message "Git 작업 중 오류가 발생했습니다."
    exit 1
fi 