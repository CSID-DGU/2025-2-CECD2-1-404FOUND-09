'use client';

import React, { useState, useEffect } from 'react';
import styled from 'styled-components';
import { Download, RefreshCw, AlertCircle, FileText, Eye } from 'lucide-react';

interface ExcelData {
  columns: string[];
  rows: any[][];
  summary?: {
    totalRows: number;
    diabetesPredicted: number;
    normalPredicted: number;
    averageProbability: number;
  };
}

const ExcelResultViewer: React.FC = () => {
  const [excelData, setExcelData] = useState<ExcelData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSummary, setShowSummary] = useState(true);
  const [retryCount, setRetryCount] = useState(0);
  const [hasTrainingCompleted, setHasTrainingCompleted] = useState(false);
  const maxRetries = 3;

  const fetchExcelData = async () => {
    // 최대 재시도 횟수 초과 시 중단
    if (retryCount >= maxRetries) {
      setError('최대 재시도 횟수를 초과했습니다. 나중에 다시 시도해주세요.');
      return;
    }

    setLoading(true);
    setError(null);
    
    try {
      // FedHybrid 로컬 학습 결과 API 사용 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10초 타임아웃
      
      const response = await fetch('/api/fedhybrid/local-training', {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        if (response.status === 404) {
          setError('아직 학습 결과 파일이 없습니다. FedHybrid-AI 모델 학습을 먼저 진행해주세요.');
        } else {
          setError('결과 파일을 불러오는데 실패했습니다.');
        }
        setRetryCount(prev => prev + 1);
        return;
      }

      const arrayBuffer = await response.arrayBuffer();
      
      // Excel 파일을 파싱하기 위해 동적으로 xlsx 라이브러리 로드
      const XLSX = await import('xlsx');
      const workbook = XLSX.read(arrayBuffer, { type: 'array' });
      
      // 첫 번째 시트 (예측결과) 읽기
      const sheetName = workbook.SheetNames[0];
      const worksheet = workbook.Sheets[sheetName];
      const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
      
      if (jsonData.length === 0) {
        setError('결과 파일이 비어있습니다.');
        return;
      }

      const columns = jsonData[0] as string[];
      const rows = jsonData.slice(1) as any[][];

      // 요약 통계 계산
      const diabetesColIndex = columns.findIndex(col => col === '예측_결과');
      const probabilityColIndex = columns.findIndex(col => col === '당뇨병_확률');
      
      let summary = undefined;
      if (diabetesColIndex !== -1 && probabilityColIndex !== -1) {
        const diabetesPredicted = rows.filter(row => row[diabetesColIndex] === 1).length;
        const normalPredicted = rows.length - diabetesPredicted;
        const probabilities = rows.map(row => row[probabilityColIndex]).filter(p => typeof p === 'number');
        const averageProbability = probabilities.length > 0 ? 
          probabilities.reduce((sum, p) => sum + p, 0) / probabilities.length : 0;

        summary = {
          totalRows: rows.length,
          diabetesPredicted,
          normalPredicted,
          averageProbability
        };
      }

      setExcelData({ columns, rows, summary });
      setRetryCount(0); // 성공 시 재시도 카운트 리셋
      
    } catch (err) {
      console.error('Excel 파싱 오류:', err);
      if (err instanceof Error && err.name === 'AbortError') {
        setError('요청 시간이 초과되었습니다. 나중에 다시 시도해주세요.');
      } else {
        setError('결과 파일을 읽는데 실패했습니다.');
      }
      setRetryCount(prev => prev + 1);
    } finally {
      setLoading(false);
    }
  };

  const downloadExcel = async () => {
    try {
      // FedHybrid 로컬 학습 결과 API 사용 (타임아웃 설정)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 다운로드는 30초 타임아웃
      
      const response = await fetch('/api/fedhybrid/local-training', {
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'prediction_results.xlsx';
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('다운로드 시간이 초과되었습니다. 나중에 다시 시도해주세요.');
      } else {
        setError('다운로드 중 오류가 발생했습니다.');
      }
    }
  };

  // 학습이 완료된 경우에만 데이터 불러오기
  useEffect(() => {
    if (hasTrainingCompleted) {
      fetchExcelData();
    }
  }, [hasTrainingCompleted]); // 학습 완료 상태가 변경될 때만 실행

  // 이벤트 리스너와 인터벌은 별도 useEffect로 분리
  useEffect(() => {
    // 학습 완료 이벤트 리스너 추가
    const handleTrainingComplete = () => {
      console.log('학습 완료 이벤트 수신, 결과 새로고침');
      setHasTrainingCompleted(true); // 학습 완료 상태 설정
      setTimeout(() => {
        fetchExcelData();
      }, 2000); // 2초 후 새로고침 (파일 생성 대기)
    };
    
    // 커스텀 이벤트 리스너 등록
    window.addEventListener('training-complete', handleTrainingComplete);
    
    return () => {
      window.removeEventListener('training-complete', handleTrainingComplete);
    };
  }, []); // 한 번만 등록

  // 주기적 체크 완전 비활성화 (성능 문제 해결을 위해)
  // useEffect(() => {
  //   if (excelData || retryCount >= maxRetries) return;
  //   const intervalId = setInterval(() => {
  //     if (!loading && retryCount < maxRetries) {
  //       fetchExcelData();
  //     }
  //   }, 30000);
  //   return () => {
  //     clearInterval(intervalId);
  //   };
  // }, [excelData, loading, retryCount]);

  if (loading) {
    return (
      <Container>
        <Header>
          <Title>AI 학습 결과</Title>
          <RefreshButton onClick={() => {
            setRetryCount(0); // 수동 새로고침 시 재시도 카운트 리셋
            if (hasTrainingCompleted) {
              fetchExcelData();
            } else {
              setError('학습이 완료된 후에 결과를 확인할 수 있습니다.');
            }
          }} disabled={loading}>
            <RefreshCw size={16} className={loading ? 'spinning' : ''} />
            새로고침
          </RefreshButton>
        </Header>
        <LoadingContainer>
          <RefreshCw size={32} className="spinning" />
          <LoadingText>결과 파일을 불러오는 중...</LoadingText>
        </LoadingContainer>
      </Container>
    );
  }

  if (error) {
    return (
      <Container>
        <Header>
          <Title>AI 학습 결과</Title>
          <RefreshButton onClick={() => {
            setRetryCount(0); // 수동 새로고침 시 재시도 카운트 리셋
            if (hasTrainingCompleted) {
              fetchExcelData();
            } else {
              setError('학습이 완료된 후에 결과를 확인할 수 있습니다.');
            }
          }}>
            <RefreshCw size={16} />
            새로고침
          </RefreshButton>
        </Header>
        <ErrorContainer>
          <AlertCircle size={48} color="#ef4444" />
          <ErrorMessage>{error}</ErrorMessage>
        </ErrorContainer>
      </Container>
    );
  }

  if (!excelData) {
    return (
      <Container>
        <Header>
          <Title>AI 학습 결과</Title>
          <RefreshButton onClick={() => {
            setRetryCount(0); // 수동 새로고침 시 재시도 카운트 리셋
            if (hasTrainingCompleted) {
              fetchExcelData();
            } else {
              setError('학습이 완료된 후에 결과를 확인할 수 있습니다.');
            }
          }}>
            <RefreshCw size={16} />
            새로고침
          </RefreshButton>
        </Header>
        <EmptyContainer>
          <FileText size={48} color="#6b7280" />
          {hasTrainingCompleted ? (
            <EmptyText>학습이 완료되었지만 결과 파일을 찾을 수 없습니다. 새로고침을 시도해보세요.</EmptyText>
          ) : (
            <EmptyText>
              FedHybrid-AI 모델 학습을 먼저 진행해주세요.<br />
              학습이 완료되면 자동으로 결과가 표시됩니다.
            </EmptyText>
          )}
        </EmptyContainer>
      </Container>
    );
  }

  return (
    <Container>
      <Header>
        <Title>AI 학습 결과</Title>
        <HeaderActions>
          <ToggleButton onClick={() => setShowSummary(!showSummary)}>
            <Eye size={16} />
            {showSummary ? '요약 숨기기' : '요약 보기'}
          </ToggleButton>
          <DownloadButton onClick={downloadExcel}>
            <Download size={16} />
            엑셀 다운로드
          </DownloadButton>
          <RefreshButton onClick={() => {
            setRetryCount(0); // 수동 새로고침 시 재시도 카운트 리셋
            if (hasTrainingCompleted) {
              fetchExcelData();
            } else {
              setError('학습이 완료된 후에 결과를 확인할 수 있습니다.');
            }
          }}>
            <RefreshCw size={16} />
            새로고침
          </RefreshButton>
        </HeaderActions>
      </Header>

      {showSummary && excelData.summary && (
        <SummarySection>
          <SummaryTitle>예측 결과 요약</SummaryTitle>
          <SummaryGrid>
            <SummaryCard>
              <SummaryLabel>총 데이터 수</SummaryLabel>
              <SummaryValue>{excelData.summary.totalRows.toLocaleString()}</SummaryValue>
            </SummaryCard>
            <SummaryCard>
              <SummaryLabel>당뇨병 예측</SummaryLabel>
              <SummaryValue danger>{excelData.summary.diabetesPredicted.toLocaleString()}</SummaryValue>
            </SummaryCard>
            <SummaryCard>
              <SummaryLabel>정상 예측</SummaryLabel>
              <SummaryValue success>{excelData.summary.normalPredicted.toLocaleString()}</SummaryValue>
            </SummaryCard>
            <SummaryCard>
              <SummaryLabel>평균 당뇨병 확률</SummaryLabel>
              <SummaryValue>{(excelData.summary.averageProbability * 100).toFixed(2)}%</SummaryValue>
            </SummaryCard>
          </SummaryGrid>
        </SummarySection>
      )}

      <ExcelTableContainer>
        <ExcelTable>
          <thead>
            <tr>
              {excelData.columns.map((column, index) => (
                <th key={index}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {excelData.rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex}>
                    {typeof cell === 'number' ? 
                      (cell > 0 && cell < 1 ? (cell * 100).toFixed(2) + '%' : cell.toString()) : 
                      (cell || '')}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </ExcelTable>
      </ExcelTableContainer>
    </Container>
  );
};

const Container = styled.div`
  width: 100%;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  margin-top: 2rem;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1.5rem 2rem;
  background: #f8fafc;
  border-bottom: 1px solid #e2e8f0;
`;

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
`;

const HeaderActions = styled.div`
  display: flex;
  gap: 0.75rem;
`;

const Button = styled.button`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  border: 1px solid #d1d5db;
  background: white;
  color: #374151;
  
  &:hover {
    background: #f3f4f6;
  }
  
  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
`;

const RefreshButton = styled(Button)`
  &:hover {
    background: #f0f9ff;
    border-color: #3b82f6;
    color: #3b82f6;
  }
`;

const DownloadButton = styled(Button)`
  background: #10b981;
  color: white;
  border-color: #10b981;
  
  &:hover {
    background: #059669;
    border-color: #059669;
  }
`;

const ToggleButton = styled(Button)`
  &:hover {
    background: #fef3c7;
    border-color: #f59e0b;
    color: #f59e0b;
  }
`;

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  gap: 1rem;
  
  .spinning {
    animation: spin 1s linear infinite;
  }
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const LoadingText = styled.p`
  color: #6b7280;
  font-size: 1rem;
  margin: 0;
`;

const ErrorContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  gap: 1rem;
`;

const ErrorMessage = styled.p`
  color: #ef4444;
  font-size: 1rem;
  text-align: center;
  margin: 0;
`;

const EmptyContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 2rem;
  gap: 1rem;
`;

const EmptyText = styled.p`
  color: #6b7280;
  font-size: 1rem;
  margin: 0;
`;

const SummarySection = styled.div`
  padding: 1.5rem 2rem;
  background: #f0fdf4;
  border-bottom: 1px solid #bbf7d0;
`;

const SummaryTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #065f46;
  margin: 0 0 1rem 0;
`;

const SummaryGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
`;

const SummaryCard = styled.div`
  background: white;
  padding: 1rem;
  border-radius: 8px;
  border: 1px solid #bbf7d0;
`;

const SummaryLabel = styled.div`
  font-size: 0.875rem;
  color: #065f46;
  margin-bottom: 0.5rem;
`;

const SummaryValue = styled.div<{ danger?: boolean; success?: boolean }>`
  font-size: 1.5rem;
  font-weight: 700;
  color: ${props => {
    if (props.danger) return '#dc2626';
    if (props.success) return '#059669';
    return '#1f2937';
  }};
`;

const ExcelTableContainer = styled.div`
  width: 100%;
  overflow-x: auto;
  max-height: 600px;
  overflow-y: auto;
`;

const ExcelTable = styled.table`
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
  
  th, td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid #e5e7eb;
    white-space: nowrap;
  }
  
  th {
    background: #f9fafb;
    font-weight: 600;
    color: #374151;
    position: sticky;
    top: 0;
    z-index: 10;
  }
  
  tr:hover {
    background: #f9fafb;
  }
  
  td {
    color: #1f2937;
  }
`;

export default ExcelResultViewer; 