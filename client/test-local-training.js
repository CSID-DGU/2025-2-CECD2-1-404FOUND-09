// 로컬 학습 기능 테스트 스크립트
// Node.js 18+ 에서는 내장 fetch 사용 가능
const fs = require('fs');
const path = require('path');

const FEDHYBRID_SERVER_URL = 'http://localhost:8000';
const CLIENT_URL = 'http://localhost:3000';

async function testLocalTraining() {
  console.log('🚀 로컬 학습 기능 테스트 시작\n');
  
  try {
    // 1. 서버 상태 확인
    console.log('1️⃣ 서버 상태 확인...');
    const statusResponse = await fetch(`${FEDHYBRID_SERVER_URL}/status`);
    if (statusResponse.ok) {
      const status = await statusResponse.json();
      console.log('✅ 서버 상태:', status.server_status);
    } else {
      console.log('❌ 서버 상태 확인 실패');
      return;
    }

    // 2. 모델 다운로드 테스트
    console.log('\n2️⃣ 모델 다운로드 테스트...');
    const modelResponse = await fetch(`${FEDHYBRID_SERVER_URL}/get_model`);
    if (modelResponse.ok) {
      const modelBuffer = await modelResponse.arrayBuffer();
      console.log('✅ 모델 다운로드 성공:', modelBuffer.byteLength, 'bytes');
    } else {
      console.log('❌ 모델 다운로드 실패');
      return;
    }

    // 3. 클라이언트 상태 확인
    console.log('\n3️⃣ 클라이언트 상태 확인...');
    try {
      const clientResponse = await fetch(`${CLIENT_URL}`);
      if (clientResponse.ok) {
        console.log('✅ 클라이언트 접속 가능');
      } else {
        console.log('❌ 클라이언트 접속 실패');
      }
    } catch (error) {
      console.log('❌ 클라이언트 연결 실패:', error.message);
    }

    // 4. 로컬 학습 API 테스트
    console.log('\n4️⃣ 로컬 학습 API 테스트...');
    try {
      const localTrainingResponse = await fetch(`${CLIENT_URL}/api/fedhybrid/local-training`);
      if (localTrainingResponse.ok) {
        console.log('✅ 로컬 학습 API 접근 가능');
      } else {
        console.log('❌ 로컬 학습 API 접근 실패:', localTrainingResponse.status);
      }
    } catch (error) {
      console.log('❌ 로컬 학습 API 연결 실패:', error.message);
    }

    console.log('\n📋 테스트 완료!');
    console.log('\n🌐 웹 브라우저에서 다음을 테스트하세요:');
    console.log('1. http://localhost:3000 접속');
    console.log('2. 참여자 대시보드로 이동');
    console.log('3. FedHybrid-AI 연동 섹션에서 파일 업로드');
    console.log('4. "모델 다운로드 & 로컬 학습 시작" 버튼 클릭');

  } catch (error) {
    console.error('❌ 테스트 중 오류 발생:', error.message);
  }
}

// 스크립트 실행
if (require.main === module) {
  testLocalTraining().catch(console.error);
}

module.exports = {
  testLocalTraining
};
