# FedHybrid-Client 배포 가이드

## 🚀 서버 배포 방법

### 방법 1: PM2를 사용한 배포 (권장)

#### 1. 서버 준비
```bash
# Node.js 설치 (18.x 이상)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# PM2 설치
npm install -g pm2

# Nginx 설치
sudo apt-get install nginx
```

#### 2. 프로젝트 업로드
```bash
# 서버에 프로젝트 폴더 생성
mkdir -p /var/www/fedhybrid-client
cd /var/www/fedhybrid-client

# 로컬에서 파일 업로드 (scp, rsync, git 등 사용)
# 예: scp -r . user@your-server:/var/www/fedhybrid-client/
```

#### 3. 환경 변수 설정
```bash
# .env.local 파일 생성
nano .env.local

# 다음 내용 추가:
NEXT_PUBLIC_SUPABASE_URL=your_actual_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_actual_supabase_key
```

#### 4. ecosystem.config.js 수정
```bash
# 실제 프로젝트 경로로 수정
nano ecosystem.config.js
# cwd: '/var/www/fedhybrid-client'로 변경
```

#### 5. 배포 실행
```bash
# 배포 스크립트 실행
./deploy.sh
```

#### 6. Nginx 설정
```bash
# nginx.conf 파일을 /etc/nginx/sites-available/에 복사
sudo cp nginx.conf /etc/nginx/sites-available/fedhybrid-client

# 도메인 설정 수정
sudo nano /etc/nginx/sites-available/fedhybrid-client

# 사이트 활성화
sudo ln -s /etc/nginx/sites-available/fedhybrid-client /etc/nginx/sites-enabled/

# Nginx 재시작
sudo systemctl restart nginx
```

### 방법 2: Docker를 사용한 배포

#### 1. Docker 설치
```bash
# Docker 설치
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Docker Compose 설치
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 2. 환경 변수 설정
```bash
# .env 파일 생성
nano .env

# 다음 내용 추가:
NEXT_PUBLIC_SUPABASE_URL=your_actual_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_actual_supabase_key
```

#### 3. Docker 배포
```bash
# Docker 이미지 빌드 및 실행
docker-compose up -d --build
```

### 방법 3: Vercel 배포 (가장 간단)

#### 1. Vercel CLI 설치
```bash
npm install -g vercel
```

#### 2. 배포
```bash
# Vercel에 로그인
vercel login

# 배포
vercel --prod
```

## 🔧 관리 명령어

### PM2 관리
```bash
# 상태 확인
pm2 status

# 로그 확인
pm2 logs fedhybrid-client

# 재시작
pm2 restart fedhybrid-client

# 중지
pm2 stop fedhybrid-client

# 삭제
pm2 delete fedhybrid-client
```

### Docker 관리
```bash
# 컨테이너 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs fedhybrid-client

# 재시작
docker-compose restart

# 중지
docker-compose down
```

## 🌐 도메인 설정

### DNS 설정
- A 레코드: `your-domain.com` → 서버 IP
- CNAME 레코드: `www.your-domain.com` → `your-domain.com`

### SSL 인증서 (Let's Encrypt)
```bash
# Certbot 설치
sudo apt-get install certbot python3-certbot-nginx

# SSL 인증서 발급
sudo certbot --nginx -d your-domain.com -d www.your-domain.com
```

## 📊 모니터링

### PM2 모니터링
```bash
# 실시간 모니터링
pm2 monit

# 웹 대시보드
pm2 web
```

### 로그 관리
```bash
# PM2 로그
pm2 logs fedhybrid-client --lines 100

# Nginx 로그
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

## 🔒 보안 설정

### 방화벽 설정
```bash
# UFW 활성화
sudo ufw enable

# 필요한 포트만 열기
sudo ufw allow ssh
sudo ufw allow 80
sudo ufw allow 443
```

### 환경 변수 보안
- `.env.local` 파일은 절대 Git에 커밋하지 마세요
- 프로덕션 환경에서는 환경 변수를 시스템 레벨에서 설정하세요

## 🚨 문제 해결

### 포트 충돌
```bash
# 포트 사용 확인
sudo netstat -tulpn | grep :3000

# 프로세스 종료
sudo kill -9 [PID]
```

### 권한 문제
```bash
# 파일 권한 수정
sudo chown -R $USER:$USER /var/www/fedhybrid-client
sudo chmod -R 755 /var/www/fedhybrid-client
```

### 메모리 부족
```bash
# 메모리 사용량 확인
free -h

# PM2 메모리 제한 설정
pm2 start ecosystem.config.js --max-memory-restart 1G
``` 

## 1. 서버에 접속해서 프로젝트 확인

```bash
# 서버 접속
ssh jyh@210.94.185.206 -p 2222

# 프로젝트 디렉토리로 이동
cd fedhybrid-client

# 파일 확인
ls -la
```

## 2. Node.js 설치 (아직 안 했다면)

```bash
# Node.js 18 설치
nvm install 18
nvm use 18
nvm alias default 18

# 설치 확인
node --version
npm --version
```

## 3. 환경 변수 파일 생성

```bash
# .env.local 파일 생성
nano .env.local
```

다음 내용을 추가하세요:
```bash
NEXT_PUBLIC_SUPABASE_URL=https://ncmphbwnecllxvwjskuv.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5jbXBobnduZWNsbHh2d2pza3V2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTE0Njg4NDIsImV4cCI6MjA2NzA0NDg0Mn0.2IGI2E-2zA0CzjsREquAp8fsTR4Aev-_kkDrf0zWyuI
```

저장: `Ctrl + X`, `Y`, `Enter`

## 4. ecosystem.config.js 수정

```bash
# 파일 수정
nano ecosystem.config.js
```

경로를 실제 경로로 수정하세요:
```javascript
module.exports = {
  apps: [
    {
      name: 'fedhybrid-client',
      script: 'npm',
      args: 'start',
      cwd: '/home/jyh/fedhybrid-client', // 실제 경로로 변경
      instances: 1,
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'production',
        PORT: 3000
      }
    }
  ]
};
```

저장: `Ctrl + X`, `Y`, `Enter`

## 5. PM2 설치

```bash
# PM2 전역 설치
npm install -g pm2

# 설치 확인
pm2 --version
```

## 6. 의존성 설치 및 빌드

```bash
# 의존성 설치
npm install

# 프로덕션 빌드
npm run build
```

## 7. 배포 실행

```bash
# 실행 권한 부여
chmod +x deploy.sh

# 배포 스크립트 실행
./deploy.sh
```

## 8. 상태 확인

```bash
# PM2 상태 확인
pm2 status

# 애플리케이션 로그 확인
pm2 logs fedhybrid-client

# 포트 사용 확인
netstat -tulpn | grep :3000
```

## 9. 접속 테스트

이제 다음 URL로 접속할 수 있습니다:
- `http://210.94.185.206:3000`

## 10. 자동 시작 설정

```bash
# PM2 시작 스크립트 생성
pm2 startup

# 현재 프로세스 저장
pm2 save
```

## 11. 모니터링

```bash
# 실시간 모니터링
pm2 monit

# 웹 대시보드 (포트 9615)
pm2 web
```

## 12. 문제 해결

만약 에러가 발생한다면:

```bash
# 로그 확인
pm2 logs fedhybrid-client --lines 50

# 프로세스 재시작
pm2 restart fedhybrid-client

# 프로세스 중지 후 다시 시작
pm2 stop fedhybrid-client
pm2 start fedhybrid-client
```

이제 단계별로 진행해보세요! 어떤 단계에서 문제가 발생하거나 결과를 확인하고 싶으시면 말씀해 주세요.

특히 다음 명령어들을 순서대로 실행해보세요:

```bash
# 1. 서버 접속
ssh jyh@210.94.185.206 -p 2222

# 2. 프로젝트 디렉토리로 이동
cd fedhybrid-client

# 3. Node.js 설치 확인
node --version
npm --version
```

이 명령어들을 실행한 후 결과를 알려주시면 다음 단계를 안내해드리겠습니다.