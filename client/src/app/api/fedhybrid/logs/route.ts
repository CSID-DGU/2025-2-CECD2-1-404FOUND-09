import { NextRequest } from 'next/server';
import { spawn } from 'child_process';
import path from 'path';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

// 실시간 로그 스트리밍을 위한 Server-Sent Events
export async function GET(request: NextRequest) {
  // Server-Sent Events 헤더 설정
  const responseHeaders = {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Headers': 'Cache-Control',
  };

  // ReadableStream을 사용한 실시간 스트리밍
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      // 연결 시작 메시지
      const startMessage = `data: ${JSON.stringify({ 
        message: '🔄 실시간 로그 스트리밍이 시작되었습니다.',
        timestamp: new Date().toISOString()
      })}\n\n`;
      controller.enqueue(encoder.encode(startMessage));

      // 주기적으로 하트비트 전송 (연결 유지)
      const heartbeatInterval = setInterval(() => {
        try {
          const heartbeat = `data: ${JSON.stringify({ 
            message: '연결 상태 확인',
            timestamp: new Date().toISOString(),
            type: 'heartbeat'
          })}\n\n`;
          controller.enqueue(encoder.encode(heartbeat));
        } catch (error) {
          console.error('하트비트 전송 실패:', error);
          clearInterval(heartbeatInterval);
        }
      }, 30000); // 30초마다 하트비트

      // 클라이언트 연결 종료 시 정리 (단일 핸들러)
      const onAbort = () => {
        clearInterval(heartbeatInterval);
        delete (global as any).sendLogToClient;
        try {
          const closeMessage = `data: ${JSON.stringify({ 
            message: '🔌 로그 스트리밍 연결이 종료되었습니다.',
            timestamp: new Date().toISOString(),
            type: 'close'
          })}\n\n`;
          controller.enqueue(encoder.encode(closeMessage));
        } catch (error) {
          console.error('스트림 종료 메시지 전송 중 오류:', error);
        }
        try {
          controller.close();
        } catch (error) {
          console.error('스트림 종료 중 오류:', error);
        }
      };
      request.signal?.addEventListener('abort', onAbort);

      // Python 프로세스 로그를 실시간으로 전달하는 함수
      const sendLogMessage = (message: string, type: string = 'log') => {
        try {
          const logMessage = `data: ${JSON.stringify({ 
            message: message,
            timestamp: new Date().toISOString(),
            type: type
          })}\n\n`;
          controller.enqueue(new TextEncoder().encode(logMessage));
        } catch (error) {
          console.error('로그 메시지 전송 실패:', error);
        }
      };

      // 전역 로그 핸들러 등록 (다른 API에서 사용할 수 있도록)
      (global as any).sendLogToClient = sendLogMessage;
    },
  });

  return new Response(stream, {
    headers: responseHeaders,
  });
}

// Python 프로세스의 실시간 출력을 스트리밍하는 함수 (향후 사용)
function streamPythonProcess(pythonScript: string, args: string[]) {
  return new ReadableStream({
    start(controller) {
      const aiDir = path.join(process.cwd(), '..', 'FedHybrid-AI');

      const pythonBin = process.env.PYTHON_BIN || 'python3';
      const pythonProcess = spawn(pythonBin, [pythonScript, ...args], {
        cwd: aiDir,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      // 표준 출력 스트리밍
      pythonProcess.stdout.on('data', (data) => {
        const output = data.toString();
        const lines = output.split('\n').filter((line: string) => line.trim());
        
        lines.forEach((line: string) => {
          try {
            const message = `data: ${JSON.stringify({ 
              message: line,
              timestamp: new Date().toISOString(),
              type: 'stdout'
            })}\n\n`;
            controller.enqueue(new TextEncoder().encode(message));
          } catch (error) {
            console.error('stdout 스트리밍 오류:', error);
          }
        });
      });

      // 표준 에러 스트리밍
      pythonProcess.stderr.on('data', (data) => {
        const output = data.toString();
        const lines = output.split('\n').filter((line: string) => line.trim());
        
        lines.forEach((line: string) => {
          try {
            const message = `data: ${JSON.stringify({ 
              message: `🚨 ${line}`,
              timestamp: new Date().toISOString(),
              type: 'stderr'
            })}\n\n`;
            controller.enqueue(new TextEncoder().encode(message));
          } catch (error) {
            console.error('stderr 스트리밍 오류:', error);
          }
        });
      });

      // 프로세스 종료 처리
      pythonProcess.on('close', (code) => {
        try {
          const message = `data: ${JSON.stringify({ 
            message: `🏁 Python 프로세스 종료 (코드: ${code})`,
            timestamp: new Date().toISOString(),
            type: 'process_exit',
            exitCode: code
          })}\n\n`;
          controller.enqueue(new TextEncoder().encode(message));
          controller.close();
        } catch (error) {
          console.error('프로세스 종료 메시지 전송 실패:', error);
        }
      });

      // 프로세스 오류 처리
      pythonProcess.on('error', (error) => {
        try {
          const message = `data: ${JSON.stringify({ 
            message: `❌ Python 프로세스 오류: ${error.message}`,
            timestamp: new Date().toISOString(),
            type: 'process_error'
          })}\n\n`;
          controller.enqueue(new TextEncoder().encode(message));
          controller.close();
        } catch (err) {
          console.error('프로세스 오류 메시지 전송 실패:', err);
        }
      });
    },
  });
}
