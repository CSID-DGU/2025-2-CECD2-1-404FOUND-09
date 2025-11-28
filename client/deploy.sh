#!/bin/bash

# 배포 스크립트
echo "FedHybrid-Client 배포 시작..."

# 1. 의존성 설치
echo "의존성 설치 중..."
npm install

# 2. 프로덕션 빌드
echo "프로덕션 빌드 중..."
npm run build

# 3. PM2로 애플리케이션 시작/재시작
echo "🔄 PM2로 애플리케이션 시작 중..."
pm2 start ecosystem.config.js --env production

echo "배포 완료!"
echo "애플리케이션이 http://localhost:3000 에서 실행 중입니다."
echo "PM2 상태 확인: pm2 status"
echo "로그 확인: pm2 logs fedhybrid-client" 