#!/bin/bash

# FedHybrid-AI와 FedHybrid-Client 연동 실행 스크립트

echo "🚀 FedHybrid 연동 시스템 시작"

# 현재 디렉토리 확인
CURRENT_DIR=$(pwd)
echo "현재 디렉토리: $CURRENT_DIR"

# FedHybrid-AI 서버 시작
echo "📡 FedHybrid-AI 서버 시작 중..."
cd ../FedHybrid-AI

# Python 가상환경 확인 및 활성화
if [ -d "venv" ]; then
    echo "가상환경 활성화 중..."
    source venv/bin/activate
fi

# 의존성 설치 확인
if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt 파일을 찾을 수 없습니다."
    exit 1
fi

echo "Python 패키지 설치 확인 중..."
pip install -r requirements.txt

# 서버 백그라운드에서 시작
echo "FedHybrid-AI 서버를 백그라운드에서 시작합니다..."
python FedHBServer.py &
FEDHYBRID_PID=$!

echo "FedHybrid-AI 서버 PID: $FEDHYBRID_PID"

# 서버 시작 대기
echo "서버 시작 대기 중..."
sleep 5

# 서버 상태 확인
echo "서버 상태 확인 중..."
curl -s http://localhost:8000/status > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ FedHybrid-AI 서버가 성공적으로 시작되었습니다."
else
    echo "❌ FedHybrid-AI 서버 시작에 실패했습니다."
    kill $FEDHYBRID_PID 2>/dev/null
    exit 1
fi

# FedHybrid-Client로 돌아가기
cd "$CURRENT_DIR"

# Node.js 의존성 설치 확인
if [ ! -f "package.json" ]; then
    echo "❌ package.json 파일을 찾을 수 없습니다."
    kill $FEDHYBRID_PID 2>/dev/null
    exit 1
fi

echo "📦 Node.js 패키지 설치 확인 중..."
npm install

# 환경 변수 파일 확인
if [ ! -f ".env.local" ]; then
    echo "⚠️  .env.local 파일이 없습니다. 생성 중..."
    cat > .env.local << EOF
# FedHybrid-AI 서버 설정
FEDHYBRID_SERVER_URL=http://localhost:8000

# Supabase 설정 (필요시 수정)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
EOF
    echo "✅ .env.local 파일이 생성되었습니다."
fi

# Next.js 개발 서버 시작
echo "🌐 FedHybrid-Client 시작 중..."
npm run dev &
CLIENT_PID=$!

echo "FedHybrid-Client PID: $CLIENT_PID"

# 서버 시작 대기
echo "클라이언트 시작 대기 중..."
sleep 10

# 클라이언트 상태 확인
echo "클라이언트 상태 확인 중..."
curl -s http://localhost:3000 > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ FedHybrid-Client가 성공적으로 시작되었습니다."
else
    echo "❌ FedHybrid-Client 시작에 실패했습니다."
    kill $CLIENT_PID 2>/dev/null
    kill $FEDHYBRID_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 FedHybrid 연동 시스템이 성공적으로 시작되었습니다!"
echo ""
echo "📱 접속 정보:"
echo "   - FedHybrid-Client: http://localhost:3000"
echo "   - FedHybrid-AI 서버: http://localhost:8000"
echo ""
echo "🛑 시스템을 종료하려면 Ctrl+C를 누르세요."

# 종료 처리 함수
cleanup() {
    echo ""
    echo "🛑 시스템 종료 중..."
    kill $CLIENT_PID 2>/dev/null
    kill $FEDHYBRID_PID 2>/dev/null
    echo "✅ 모든 프로세스가 종료되었습니다."
    exit 0
}

# 시그널 핸들러 설정
trap cleanup SIGINT SIGTERM

# 프로세스 모니터링
while true; do
    if ! kill -0 $FEDHYBRID_PID 2>/dev/null; then
        echo "❌ FedHybrid-AI 서버가 종료되었습니다."
        cleanup
    fi
    
    if ! kill -0 $CLIENT_PID 2>/dev/null; then
        echo "❌ FedHybrid-Client가 종료되었습니다."
        cleanup
    fi
    
    sleep 5
done
