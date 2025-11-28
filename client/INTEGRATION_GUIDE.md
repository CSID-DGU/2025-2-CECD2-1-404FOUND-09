# FedHybrid 연동 가이드

이 가이드는 FedHybrid-AI와 FedHybrid-Client를 연동하여 사용하는 방법을 설명합니다.

## 🚀 빠른 시작

### 1. 자동 실행 (권장)

```bash
# FedHybrid-Client 디렉토리에서
./start-integration.sh
```

이 스크립트는 다음을 자동으로 수행합니다:
- FedHybrid-AI 서버 시작
- FedHybrid-Client 시작
- 환경 변수 설정
- 의존성 설치 확인

### 2. 수동 실행

#### FedHybrid-AI 서버 시작
```bash
cd ../FedHybrid-AI
python FedHBServer.py
```

#### FedHybrid-Client 시작
```bash
cd FedHybrid-Client
npm install
npm run dev
```

## 📋 사전 요구사항

### FedHybrid-AI
- Python 3.8+
- 필요한 패키지: `pip install -r requirements.txt`
- 데이터 파일: `diabetic_data.csv`

### FedHybrid-Client
- Node.js 18+
- npm 또는 yarn

## ⚙️ 환경 설정

### 환경 변수 설정
`.env.local` 파일을 생성하고 다음 내용을 추가하세요:

```env
# FedHybrid-AI 서버 설정
FEDHYBRID_SERVER_URL=http://localhost:8000

# Supabase 설정 (필요시)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

## 🔧 연동 기능

### 1. 서버 상태 확인
- FedHybrid-AI 서버의 현재 상태 확인
- 학습 진행 상황 모니터링
- 활성 라운드 및 클라이언트 수 확인

### 2. 모델 다운로드
- 서버에서 글로벌 모델 다운로드
- 로컬 학습을 위한 모델 준비

### 3. 로컬 학습
- 다운로드한 모델로 로컬에서 학습 진행
- FedHybrid-AI 클라이언트를 통한 학습 수행
- 학습 진행 상황 실시간 모니터링

### 4. 학습 결과 다운로드
- 로컬 학습 완료 후 예측 결과 Excel 파일 다운로드
- 학습된 모델 다운로드
- 성능 지표 확인

## 📊 API 엔드포인트

### FedHybrid-Client API
- `GET /api/fedhybrid?action=status` - 서버 상태 확인
- `GET /api/fedhybrid?action=get_model` - 모델 다운로드
- `POST /api/fedhybrid/local-training` - 로컬 학습 시작
- `GET /api/fedhybrid/local-training` - 로컬 학습 결과 다운로드
- `POST /api/fedhybrid?action=update_config` - 설정 업데이트

### FedHybrid-AI API
- `GET /status` - 서버 상태
- `GET /predict_and_download` - 예측 결과 다운로드
- `GET /get_model` - 모델 다운로드
- `POST /upload_data` - 데이터 업로드
- `POST /config` - 설정 업데이트
- `POST /aggregate` - 모델 집계

## 🧪 테스트

### 연동 테스트 실행
```bash
node test-integration.js
```

### 로컬 학습 테스트 실행
```bash
node test-local-training.js
```

이 테스트는 다음을 확인합니다:
- 서버 연결 상태
- 모델 다운로드 기능
- 로컬 학습 API 접근
- 클라이언트 연결 상태

## 🔍 문제 해결

### 일반적인 문제

#### 1. CORS 오류
- FedHybrid-AI 서버의 CORS 설정 확인
- 클라이언트 URL이 허용 목록에 포함되어 있는지 확인

#### 2. 서버 연결 실패
- FedHybrid-AI 서버가 실행 중인지 확인
- 포트 8000이 사용 가능한지 확인
- 방화벽 설정 확인

#### 3. 파일 업로드 실패
- 파일 형식이 지원되는지 확인 (CSV, Excel)
- 파일 크기 제한 확인
- 서버 디스크 공간 확인

#### 4. 모델 다운로드 실패
- 학습이 완료되었는지 확인
- 모델 파일이 존재하는지 확인
- 서버 로그 확인

### 로그 확인

#### FedHybrid-AI 서버 로그
```bash
# 서버 실행 시 콘솔에 출력되는 로그 확인
python FedHBServer.py
```

#### FedHybrid-Client 로그
```bash
# 브라우저 개발자 도구에서 네트워크 탭 확인
# 또는 터미널에서 Next.js 로그 확인
npm run dev
```

## 📈 성능 최적화

### 1. 서버 설정 최적화
- `ROUND_CONFIG` 설정 조정
- 타임아웃 값 최적화
- 클라이언트 수 조정

### 2. 클라이언트 설정 최적화
- 파일 업로드 크기 제한
- 요청 타임아웃 설정
- 에러 재시도 로직

## 🔐 보안 고려사항

### 1. 인증
- Supabase 인증 사용
- API 엔드포인트 보호
- 사용자 권한 확인

### 2. 데이터 보안
- CKKS 암호화 사용
- 안전한 파일 업로드
- 민감한 데이터 보호

## 📚 추가 리소스

- [FedHybrid-AI README](../FedHybrid-AI/README.md)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Next.js 문서](https://nextjs.org/docs)
- [CKKS 암호화](https://en.wikipedia.org/wiki/Homomorphic_encryption)

## 🤝 기여하기

버그 리포트나 기능 요청은 이슈를 통해 제출해주세요.

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.
