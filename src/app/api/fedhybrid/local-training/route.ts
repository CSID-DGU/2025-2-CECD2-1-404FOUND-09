import { NextRequest, NextResponse } from 'next/server';
import { createClient } from '@/utils/supabase/server';
import { cookies } from 'next/headers';
import { spawn } from 'child_process';
import path from 'path';
import * as XLSX from 'xlsx';
import fs from 'fs';

// Excel 파일을 CSV로 변환하는 함수
function convertExcelToCSV(filePath: string): string {
  try {
    console.log('Excel 파일 변환 시작:', filePath);
    console.log('파일 존재 여부:', fs.existsSync(filePath));
    console.log('파일 크기:', fs.existsSync(filePath) ? fs.statSync(filePath).size : 'N/A');
    
    if (!fs.existsSync(filePath)) {
      throw new Error(`파일이 존재하지 않습니다: ${filePath}`);
    }
    
    // 파일 읽기 권한 확인
    try {
      fs.accessSync(filePath, fs.constants.R_OK);
    } catch (accessError) {
      throw new Error(`파일 읽기 권한이 없습니다: ${filePath}`);
    }
    
    // 파일을 버퍼로 읽어서 처리
    const fileBuffer = fs.readFileSync(filePath);
    const workbook = XLSX.read(fileBuffer, { type: 'buffer' });
    const sheetName = workbook.SheetNames[0]; // 첫 번째 시트 사용
    const worksheet = workbook.Sheets[sheetName];
    const csvData = XLSX.utils.sheet_to_csv(worksheet);
    console.log('Excel 파일 변환 완료, CSV 데이터 크기:', csvData.length);
    return csvData;
  } catch (error) {
    console.error('Excel 파일 변환 오류:', error);
    throw new Error(`Excel 파일을 CSV로 변환할 수 없습니다: ${error}`);
  }
}

export async function POST(request: NextRequest) {
  try {
    // 개발 환경에서는 인증 우회
    // TODO: 프로덕션 환경에서는 적절한 인증 구현 필요

    const formData = await request.formData();
    const file = formData.get('file') as File;
    
    if (!file) {
      return NextResponse.json({ error: '파일이 필요합니다.' }, { status: 400 });
    }

    // 파일 확장자 확인
    const fileName = file.name;
    const fileExtension = path.extname(fileName).toLowerCase();
    
    if (!['.csv', '.xlsx', '.xls'].includes(fileExtension)) {
      return NextResponse.json({ 
        error: 'CSV 또는 Excel 파일(.xlsx, .xls)만 지원됩니다.' 
      }, { status: 400 });
    }

    // 예측 결과 파일 업로드 방지
    if (fileName.includes('diabetic_predictions') || fileName.includes('prediction_results')) {
      return NextResponse.json({ 
        error: '예측 결과 파일은 업로드할 수 없습니다. 원본 데이터 파일만 업로드해주세요.' 
      }, { status: 400 });
    }

    // 업로드 디렉토리 생성
    const uploadDir = path.join(process.cwd(), 'uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }

    // 파일 저장
    const originalFilePath = path.join(uploadDir, fileName);
    const bytes = await file.arrayBuffer();
    const buffer = Buffer.from(bytes);
    fs.writeFileSync(originalFilePath, buffer);
    console.log('원본 파일 저장 완료:', originalFilePath, '크기:', buffer.length);

    // Excel 파일인 경우 CSV로 변환
    let finalFilePath = originalFilePath;
    if (['.xlsx', '.xls'].includes(fileExtension)) {
      try {
        console.log('Excel 파일을 CSV로 변환 중...');
        
        // 파일이 완전히 저장될 때까지 잠시 대기
        await new Promise(resolve => setTimeout(resolve, 100));
        
        const csvData = convertExcelToCSV(originalFilePath);
        const csvFileName = fileName.replace(fileExtension, '.csv');
        finalFilePath = path.join(uploadDir, csvFileName);
        fs.writeFileSync(finalFilePath, csvData, 'utf8');
        console.log('Excel 파일 변환 완료:', csvFileName);
      } catch (error) {
        console.error('Excel 변환 실패:', error);
        // 변환 실패 시 오류 반환
        return NextResponse.json({ 
          error: 'Excel 파일을 CSV로 변환할 수 없습니다. 파일이 손상되었거나 지원되지 않는 형식일 수 있습니다.',
          details: error instanceof Error ? error.message : String(error)
        }, { status: 400 });
      }
    }

    // FedHybrid-AI 클라이언트 실행 (FedHBClient.py 사용)
    // 경로 수정: client/FedHybrid-Client -> ai/FedHybrid-AI
    const aiDir = path.resolve(process.cwd(), '..', '..', 'ai', 'FedHybrid-AI');
    const pythonScript = path.join(aiDir, 'FedHBClient.py');

    console.log('FedHybrid-AI 디렉토리:', aiDir);
    console.log('Python 스크립트:', pythonScript);
    console.log('입력 파일:', finalFilePath);

    // 디렉토리와 파일 존재 확인
    if (!fs.existsSync(aiDir)) {
      console.error('FedHybrid-AI 디렉토리가 존재하지 않습니다:', aiDir);
      return NextResponse.json({ error: 'FedHybrid-AI 디렉토리를 찾을 수 없습니다.' }, { status: 500 });
    }

    if (!fs.existsSync(pythonScript)) {
      console.error('Python 스크립트가 존재하지 않습니다:', pythonScript);
      return NextResponse.json({ error: 'FedHBClient.py 파일을 찾을 수 없습니다.' }, { status: 500 });
    }

    // 업로드된 파일을 FedHybrid-AI 디렉토리로 복사 (diabetic_data.csv 또는 --input_file 인자로 사용)
    const targetDataPath = path.join(aiDir, 'diabetic_data.csv');
    try {
      fs.copyFileSync(finalFilePath, targetDataPath);
      console.log('데이터 파일 복사 완료:', targetDataPath);
    } catch (copyError) {
      console.error('파일 복사 실패:', copyError);
      return NextResponse.json({ 
        error: '데이터 파일을 복사할 수 없습니다.',
        details: copyError instanceof Error ? copyError.message : String(copyError)
      }, { status: 500 });
    }

    // Python 프로세스 실행
    const pythonBin = process.env.PYTHON_BIN || 'python3';
    console.log('Python 실행 파일:', pythonBin);
    console.log('FedHBClient.py 실행 시작...');

    const pythonProcess = spawn(pythonBin, [pythonScript, '--input_file', targetDataPath], {
      cwd: aiDir,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1', // Python 출력 버퍼링 비활성화 (실시간 로그)
      }
    });

    // 표준 출력을 실시간으로 전역 로그 핸들러로 전달
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('[Python stdout]', output);
      
      // 전역 로그 핸들러가 있으면 실시간으로 전달
      if ((global as any).sendLogToClient) {
        const lines = output.split('\n').filter((line: string) => line.trim());
        lines.forEach((line: string) => {
          (global as any).sendLogToClient(line, 'python_output');
        });
      }
    });

    // 표준 에러를 실시간으로 전역 로그 핸들러로 전달
    pythonProcess.stderr.on('data', (data) => {
      const output = data.toString();
      console.error('[Python stderr]', output);
      
      // 전역 로그 핸들러가 있으면 실시간으로 전달
      if ((global as any).sendLogToClient) {
        const lines = output.split('\n').filter((line: string) => line.trim());
        lines.forEach((line: string) => {
          (global as any).sendLogToClient(line, 'python_error');
        });
      }
    });

    // 프로세스 종료 처리
    pythonProcess.on('close', (code) => {
      console.log(`Python 프로세스 종료: 코드 ${code}`);
      
      if ((global as any).sendLogToClient) {
        if (code === 0) {
          (global as any).sendLogToClient('✅ FedHybrid 클라이언트 학습이 성공적으로 완료되었습니다!', 'success');
          (global as any).sendLogToClient('📊 예측 결과 파일이 생성되었습니다.', 'success');
        } else {
          (global as any).sendLogToClient(`❌ 학습이 오류로 종료되었습니다 (코드: ${code})`, 'error');
        }
      }
    });

    // 프로세스 오류 처리
    pythonProcess.on('error', (error) => {
      console.error('Python 프로세스 오류:', error);
      if ((global as any).sendLogToClient) {
        (global as any).sendLogToClient(`❌ Python 프로세스 오류: ${error.message}`, 'error');
      }
    });

    // 즉시 응답 반환 (프로세스는 백그라운드에서 계속 실행)
    return NextResponse.json({ 
      success: true, 
      message: 'FedHybrid 클라이언트 학습이 시작되었습니다. 실시간 로그를 확인하세요.',
      processId: pythonProcess.pid,
      fileName: fileName,
      filePath: finalFilePath,
      targetPath: targetDataPath
    });

  } catch (error) {
    console.error('Local training API error:', error);
    return NextResponse.json({ error: '서버 오류가 발생했습니다.' }, { status: 500 });
  }
}

export async function GET() {
  try {
    // 개발 환경에서는 인증 우회
    // TODO: 프로덕션 환경에서는 적절한 인증 구현 필요

    // 결과 파일 확인
    // 경로 수정: client/FedHybrid-Client -> ai/FedHybrid-AI
    const aiDir = path.resolve(process.cwd(), '..', '..', 'ai', 'FedHybrid-AI');
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
