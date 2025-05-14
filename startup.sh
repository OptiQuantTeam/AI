#!/bin/bash

# 프로젝트 디렉토리로 이동
cd /workspace

# AI_Lambda 디렉토리 생성 및 이동
mkdir -p AI_Lambda
cd AI_Lambda

# AI_Lambda 레포지토리 클론
git clone https://github.com/OptiQuantTeam/AI_Lambda.git .

# 상위 디렉토리로 이동하여 AI 학습 실행
cd ..
python3 src/main2.py

# 학습된 모델 파일을 AI_Lambda/model 디렉토리로 복사
cp -r saved_models/* AI_Lambda/model/

# AI_Lambda 디렉토리로 이동
cd AI_Lambda

# 변경사항 커밋 및 푸시
if git add . && git commit -m "Update model files" && git push; then
    echo "모델 파일이 성공적으로 업로드되었습니다."
    # 서버 종료
    sudo shutdown -h now
else
    echo "Git 작업 중 오류가 발생했습니다."
    exit 1
fi 