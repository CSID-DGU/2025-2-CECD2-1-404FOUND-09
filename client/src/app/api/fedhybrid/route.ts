import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { cookies } from 'next/headers';

// FedHybrid-AI 서버 설정 (기본 포트를 8082로 변경)
const FEDHYBRID_SERVER_URL = process.env.FEDHYBRID_SERVER_URL || 'http://localhost:8082';

export async function GET(request: NextRequest) {
  try {
    // 개발 환경에서는 인증 우회
    // TODO: 프로덕션 환경에서는 적절한 인증 구현 필요

    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');

    switch (action) {
      case 'status':
        return await getServerStatus();
      case 'download_predictions':
        return await downloadPredictions();
      case 'get_model':
        return await getModel();
      default:
        return NextResponse.json({ error: '잘못된 액션입니다.' }, { status: 400 });
    }

  } catch (error) {
    console.error('FedHybrid API error:', error);
    return NextResponse.json({ error: '서버 오류가 발생했습니다.' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    // 개발 환경에서는 인증 우회
    // TODO: 프로덕션 환경에서는 적절한 인증 구현 필요

    const { searchParams } = new URL(request.url);
    const action = searchParams.get('action');

    switch (action) {
      case 'start_training':
        return await startTraining(request);
      case 'update_config':
        return await updateConfig(request);
      default:
        return NextResponse.json({ error: '잘못된 액션입니다.' }, { status: 400 });
    }

  } catch (error) {
    console.error('FedHybrid API error:', error);
    return NextResponse.json({ error: '서버 오류가 발생했습니다.' }, { status: 500 });
  }
}

// 서버 상태 확인
async function getServerStatus() {
  try {
    const response = await fetch(`${FEDHYBRID_SERVER_URL}/status`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('서버 상태 확인 실패:', error);
    return NextResponse.json({ 
      error: 'FedHybrid-AI 서버에 연결할 수 없습니다.',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 503 });
  }
}

// 예측 결과 다운로드
async function downloadPredictions() {
  try {
    const response = await fetch(`${FEDHYBRID_SERVER_URL}/predict_and_download`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const buffer = await response.arrayBuffer();
    
    return new NextResponse(buffer, {
      headers: {
        'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'Content-Disposition': 'attachment; filename="diabetic_predictions.xlsx"'
      }
    });
  } catch (error) {
    console.error('예측 결과 다운로드 실패:', error);
    return NextResponse.json({ 
      error: '예측 결과를 다운로드할 수 없습니다.',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// 모델 다운로드
async function getModel() {
  try {
    const response = await fetch(`${FEDHYBRID_SERVER_URL}/get_model`, {
      method: 'GET',
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const buffer = await response.arrayBuffer();
    
    return new NextResponse(buffer, {
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': 'attachment; filename="global_model.pth"'
      }
    });
  } catch (error) {
    console.error('모델 다운로드 실패:', error);
    return NextResponse.json({ 
      error: '모델을 다운로드할 수 없습니다.',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// 학습 시작
async function startTraining(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: '파일이 필요합니다.' }, { status: 400 });
    }

    // 파일을 FedHybrid-AI 서버로 전송
    const uploadFormData = new FormData();
    uploadFormData.append('file', file);

    const response = await fetch(`${FEDHYBRID_SERVER_URL}/upload_data`, {
      method: 'POST',
      body: uploadFormData,
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('학습 시작 실패:', error);
    return NextResponse.json({ 
      error: '학습을 시작할 수 없습니다.',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}

// 설정 업데이트
async function updateConfig(request: NextRequest) {
  try {
    const body = await request.json();
    
    const response = await fetch(`${FEDHYBRID_SERVER_URL}/config`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('설정 업데이트 실패:', error);
    return NextResponse.json({ 
      error: '설정을 업데이트할 수 없습니다.',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}
