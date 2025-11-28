// FedHybrid-AI 서버 연동 테스트 스크립트
// Node.js 18+ 에서는 내장 fetch 사용 가능

const FEDHYBRID_SERVER_URL = 'http://localhost:8000';

async function testServerStatus() {
  console.log('🔍 서버 상태 확인 중...');
  
  try {
    const response = await fetch(`${FEDHYBRID_SERVER_URL}/status`);
    if (response.ok) {
      const data = await response.json();
      console.log('✅ 서버 상태:', data);
      return true;
    } else {
      console.log('❌ 서버 응답 오류:', response.status);
      return false;
    }
  } catch (error) {
    console.log('❌ 서버 연결 실패:', error.message);
    return false;
  }
}

async function testModelDownload() {
  console.log('📥 모델 다운로드 테스트 중...');
  
  try {
    const response = await fetch(`${FEDHYBRID_SERVER_URL}/get_model`);
    if (response.ok) {
      const arrayBuffer = await response.arrayBuffer();
      console.log('✅ 모델 다운로드 성공:', arrayBuffer.byteLength, 'bytes');
      return true;
    } else {
      console.log('❌ 모델 다운로드 실패:', response.status);
      return false;
    }
  } catch (error) {
    console.log('❌ 모델 다운로드 오류:', error.message);
    return false;
  }
}

async function testPredictionsDownload() {
  console.log('📊 예측 결과 다운로드 테스트 중...');
  
  try {
    const response = await fetch(`${FEDHYBRID_SERVER_URL}/predict_and_download`);
    if (response.ok) {
      const arrayBuffer = await response.arrayBuffer();
      console.log('✅ 예측 결과 다운로드 성공:', arrayBuffer.byteLength, 'bytes');
      return true;
    } else {
      console.log('❌ 예측 결과 다운로드 실패:', response.status);
      return false;
    }
  } catch (error) {
    console.log('❌ 예측 결과 다운로드 오류:', error.message);
    return false;
  }
}

async function runTests() {
  console.log('🚀 FedHybrid-AI 서버 연동 테스트 시작\n');
  
  const tests = [
    { name: '서버 상태', test: testServerStatus },
    { name: '모델 다운로드', test: testModelDownload },
    { name: '예측 결과 다운로드', test: testPredictionsDownload }
  ];
  
  let passed = 0;
  let total = tests.length;
  
  for (const { name, test } of tests) {
    console.log(`\n--- ${name} 테스트 ---`);
    const result = await test();
    if (result) passed++;
    console.log(`--- ${name} 테스트 완료 ---\n`);
  }
  
  console.log(`\n📋 테스트 결과: ${passed}/${total} 통과`);
  
  if (passed === total) {
    console.log('🎉 모든 테스트가 성공했습니다!');
  } else {
    console.log('⚠️  일부 테스트가 실패했습니다.');
  }
}

// 스크립트 실행
if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = {
  testServerStatus,
  testModelDownload,
  testPredictionsDownload,
  runTests
};
