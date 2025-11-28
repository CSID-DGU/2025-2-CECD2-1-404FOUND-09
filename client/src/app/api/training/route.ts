import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { cookies } from 'next/headers';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

export async function POST(request: NextRequest) {
  try {
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    
    // 사용자 인증 확인
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    if (authError || !user) {
      return NextResponse.json({ error: '인증이 필요합니다.' }, { status: 401 });
    }

    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: '파일이 필요합니다.' }, { status: 400 });
    }

    // 파일 확장자 확인
    const fileName = file.name;
    const fileExtension = path.extname(fileName).toLowerCase();
    
    if (!['.csv', '.xlsx', '.xls'].includes(fileExtension)) {
      return NextResponse.json({ error: 'CSV 또는 Excel 파일만 업로드 가능합니다.' }, { status: 400 });
    }

    // 업로드 디렉토리 생성
    const uploadDir = path.join(process.cwd(), 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    // 파일 저장
    const filePath = path.join(uploadDir, fileName);
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    fs.writeFileSync(filePath, buffer);

    // AI 모델 실행 (FedHBClient.py 사용)
    const aiDir = path.join(process.cwd(), '..', 'FedHybrid-AI');
    const pythonScript = path.join(aiDir, 'FedHBClient.py');

    // 간단한 응답으로 변경 (실제 Python 프로세스 실행은 별도로 처리)
    return NextResponse.json({ 
      success: true, 
      message: '파일이 성공적으로 업로드되었습니다.',
      fileName: fileName,
      filePath: filePath
    });

  } catch (error) {
    console.error('Training API error:', error);
    return NextResponse.json({ error: '서버 오류가 발생했습니다.' }, { status: 500 });
  }
}

export async function GET() {
  try {
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    
    // 사용자 인증 확인
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    if (authError || !user) {
      return NextResponse.json({ error: '인증이 필요합니다.' }, { status: 401 });
    }

    // 결과 파일 확인
    const aiDir = path.join(process.cwd(), '..', 'FedHybrid-AI');
    const resultPath = path.join(aiDir, 'prediction_results.xlsx');
    
    if (fs.existsSync(resultPath)) {
      const fileBuffer = fs.readFileSync(resultPath);
      return new NextResponse(fileBuffer, {
        headers: {
          'Content-Type': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
          'Content-Disposition': 'attachment; filename="prediction_results.xlsx"'
        }
      });
    } else {
      return NextResponse.json({ error: '결과 파일을 찾을 수 없습니다.' }, { status: 404 });
    }

  } catch (error) {
    console.error('Download API error:', error);
    return NextResponse.json({ error: '서버 오류가 발생했습니다.' }, { status: 500 });
  }
} 