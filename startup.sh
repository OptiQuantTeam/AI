#!/bin/bash

# 프로젝트 디렉토리로 이동
cd /workspace

# Python 스크립트 실행
python3 main2.py

# 스크립트가 종료되면 Git 작업 수행
if git add . && git commit -m "Auto commit on script completion" && git push; then
    echo "Git 작업이 성공적으로 완료되었습니다."
    # 서버 종료
    sudo shutdown -h now
else
    echo "Git 작업 중 오류가 발생했습니다."
    exit 1
fi 