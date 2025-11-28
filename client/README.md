<div align="center">
  
<img src="https://capsule-render.vercel.app/api?type=waving&color=auto&height=200&section=header&text=FedHybrid&fontSize=56" width="100%"/>

</div>

## ✨ 프로젝트 소개

```
FedHybrid는 Federated Learning + Hybrid 에서 유래하여, 기존 연합 학습 (FL) 분야의
이론들을 통합하여 상용화 목적에 부합하는 최적의 AI 연합 학습 환경 개발을 목표로 합니다.
```

[![Watch the video](https://github.com/user-attachments/assets/c15441e0-71f6-49f0-807d-ed69e968725e)](https://youtu.be/MOwfVM7K7Wc?si=TjV8VLWX4CYSCrjR)
> 화면을 클릭하시면 프로젝트 소개 영상으로 넘어갑니다!

<br/>

## 👋 팀원 소개

| 이름                                        | 전공        | 역할 | Email                |
| -------------------------------------------- | -------------- | ------ | -------------------- |
| [정영호](https://github.com/Young-Ho-Boss) | 융합보안학과 | AI | dudgh0314@naver.com |
| [김수현](https://github.com/gitsuhyun) | 컴퓨터공학전공 | FE | suhyun10201@naver.com |
| [김지우](https://github.com/ryann1203) | 컴퓨터공학전공 | FE | 2022110421@dgu.ac.kr |
| [장재혁](https://github.com/34-43) | 컴퓨터공학전공 | BE | jehyuck3443@gmail.com |

<br/>

## 🌟 프로젝트 주요 기능

| 기능 | 설명 |
| --- | --- |
| **데이터 업로드** | CSV, XLSX 형식의 데이터 파일을 업로드하여 학습을 진행합니다. |
| **실시간 학습 로그 및 정확도 차트 확인** | AI 서버에서 진행되는 학습 과정을 실시간으로 시각화하여 제공합니다. |
| **결과 요약** | 학습 종료 시, 평균 정확도 및 예측 결과를 요약하여 제공합니다. |
| **결과 엑셀 파일 시각화 및 다운로드** | 학습 완료 후 예측 결과 엑셀 데이터를 생성합니다. 화면에서 즉시 확인 가능하며, 엑셀 파일로도 다운로드할 수 있습니다. |

<br/>

## 📸 화면 캡쳐

| 메인 페이지 | 소개 페이지 |
|:---:|:---:|
|<img width="1464" height="832" src="https://github.com/user-attachments/assets/13d3c250-8ff9-4d6c-aa55-be260775a32b" />|<img width="1464" height="832"  src="https://github.com/user-attachments/assets/4a9791a1-d91e-4cee-92f4-9ff82cda849c" />|
<br/>

<div align="center">

| 대시보드 (데이터 업로드) |
|:---:|
|<img src="https://github.com/user-attachments/assets/e34892e7-49d8-41d9-a990-32dd8e451059" width="100%" align="center">|

</div>

<div align="center">

| 대시보드 (결과 확인 및 다운로드) |
|:---:|
|<img src="https://github.com/user-attachments/assets/b1446c58-68d9-4ae0-af96-b7d32cbc2cb8" width="100%" align="center">|

</div>

<div align="center">
  
| 인스턴스 리스트 |
|:---:|
|<img src="https://github.com/user-attachments/assets/2c7cf130-b27a-41bd-a693-e208c8853313" width="100%" align="center">|

</div>
<br/>

## 🧩 ARCHITECTURE

### 1. 전체 아키텍쳐 구조
<img width="1920" height="1080" src="https://github.com/user-attachments/assets/34c9570e-ced9-4aa1-9fe0-9e999a4664d3" />

### 2. AI 아키텍쳐 구조
<img width="1920" height="1080" src="https://github.com/user-attachments/assets/b4f1ed5c-f170-4ad2-b0a8-f3630f9e53b2" />

<br/>

## 🔗 FedHybrid-AI 연동 설정

### 1. 환경 변수 설정

프로젝트 루트에 `.env.local` 파일을 생성하고 다음 내용을 추가하세요:

```env
# FedHybrid-AI 서버 설정
FEDHYBRID_SERVER_URL=http://localhost:8000

# Supabase 설정 (기존 설정이 있다면 유지)
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_supabase_anon_key
```

### 2. FedHybrid-AI 서버 실행

FedHybrid-AI 디렉토리에서 서버를 실행하세요:

```bash
cd ../FedHybrid-AI
python FedHBServer.py
```

서버는 `http://localhost:8000`에서 실행됩니다.

### 3. 연동 기능

- **서버 상태 확인**: FedHybrid-AI 서버의 현재 상태와 학습 진행 상황을 확인할 수 있습니다.
- **데이터 업로드**: CSV 또는 Excel 파일을 업로드하여 학습을 시작할 수 있습니다.
- **예측 결과 다운로드**: 학습 완료 후 예측 결과를 Excel 파일로 다운로드할 수 있습니다.
- **모델 다운로드**: 학습된 모델을 다운로드할 수 있습니다.

<br/>

## 🤖 설치 및 실행

```bash
npm install
npm run dev
```

애플리케이션은 `http://localhost:3000`에서 실행됩니다.

<br/>

## 🖥️ 기술 스택
- Next.js
- TypeScript
- styled-components
- ESLint
- Prettier
- Supabase

<br/>

## 📦 프로젝트 구조
```
├📂 src
├──📂 app
│   ├──📂 api # Next 서버 api 구현
│   └──📂 screens # 각 화면을 구성하는 폴더, screens로 네이밍 대체
├──📂 components
├──📂 constants
├──📂 lib
├──📂 styles
├──📂 types
├──📂 utils
└📂 upload # AI 학습 업로드 테스트 파일
```  

<br/>

## 📌 CONVENTION

- [ Git Convention ](https://colossal-empress-cc5.notion.site/Git-Convention-24a6de7fa5b780c49e53c4944d4535f2?pvs=74)

<br/>
