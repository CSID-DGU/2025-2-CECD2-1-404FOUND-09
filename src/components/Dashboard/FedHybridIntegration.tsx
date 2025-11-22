"use client";

import React, { useState, useEffect, useRef } from "react";
import styled from "styled-components";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface ServerStatus {
  current_round: number;
  active_rounds: Record<string, any>;
  global_accuracies: number[];
  round_config: {
    min_clients: number;
    max_clients: number;
    target_clients: number;
  };
}

interface FedHybridIntegrationProps {
  className?: string;
}

export default function FedHybridIntegration({
  className = "",
}: FedHybridIntegrationProps) {
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<string>("");
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [showLogs, setShowLogs] = useState(true); // 기본적으로 로그 표시
  const [currentLog, setCurrentLog] = useState<string>("");
  const [autoScroll, setAutoScroll] = useState(true); // 자동 스크롤 상태 (맨 처음만 사용)
  const maxLogs = 50; // 최대 로그 개수 제한
  const [accuracyData, setAccuracyData] = useState<{round: number; accuracy: number}[]>([]); // 정확도 차트 데이터
  const lastRoundRef = useRef<number>(0);

  // 서버 상태 확인 (타임아웃 및 에러 처리 개선)
  const checkServerStatus = async () => {
    setIsLoading(true);
    setError(null);

    try {
      // 타임아웃 설정 (5초)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);

      const response = await fetch("/api/fedhybrid?action=status", {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error("서버 상태를 확인할 수 없습니다.");
      }

      const data = await response.json();
      setServerStatus(data);
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        setError(
          "서버 응답 시간이 초과되었습니다. FedHybrid-AI 서버가 실행 중인지 확인해주세요."
        );
      } else {
        setError(
          err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다."
        );
      }
      console.error("서버 상태 확인 실패:", err);
    } finally {
      setIsLoading(false);
    }
  };

  // 파일 업로드 처리
  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      // 파일 형식 검증
      const allowedTypes = [
        "text/csv",
        "application/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
      ];
      const fileExtension = file.name.toLowerCase().split(".").pop();

      if (
        !allowedTypes.includes(file.type) &&
        !["csv", "xlsx", "xls"].includes(fileExtension || "")
      ) {
        setError("CSV 또는 Excel 파일(.xlsx, .xls)만 업로드 가능합니다.");
        event.target.value = "";
        setUploadedFile(null);
        return;
      }

      // 예측 결과 파일 업로드 방지
      if (
        file.name.includes("diabetic_predictions") ||
        file.name.includes("prediction_results")
      ) {
        setError(
          "예측 결과 파일은 업로드할 수 없습니다. 원본 데이터 파일만 업로드해주세요."
        );
        event.target.value = "";
        setUploadedFile(null);
        return;
      }

      setUploadedFile(file);
      setError(null);
    }
  };

  // 실시간 로그 업데이트 함수 (성능 최적화)
  const updateTrainingLog = (message: string) => {
    // ROUND_INFO JSON 파싱 (우선 처리, 로그에 추가하지 않음)
    if (message.includes("ROUND_INFO:")) {
      try {
        const jsonStr = message.split("ROUND_INFO:")[1]?.trim();
        if (jsonStr) {
          const roundInfo = JSON.parse(jsonStr);
          const { round, accuracy_after, accuracy_before, loss, duration } = roundInfo;
          
          // 라운드 정보 업데이트
          lastRoundRef.current = round;
          
          // 정확도 데이터 업데이트 (학습 후 정확도 사용)
          setAccuracyData((prev) => {
            // 같은 라운드가 있으면 업데이트, 없으면 추가
            const existingIndex = prev.findIndex((d) => d.round === round);
            const newData = { round, accuracy: accuracy_after };
            
            if (existingIndex >= 0) {
              const updated = [...prev];
              updated[existingIndex] = newData;
              return updated;
            } else {
              return [...prev, newData].sort((a, b) => a.round - b.round);
            }
          });
        }
      } catch (error) {
        console.error("ROUND_INFO 파싱 오류:", error, message);
      }
      // ROUND_INFO는 로그에 추가하지 않음 (차트 데이터만 업데이트)
      return;
    }

    const timestamp = new Date().toLocaleTimeString();
    const logEntry = `[${timestamp}] ${message}`;

    setTrainingLogs((prev) => {
      const newLogs = [...prev, logEntry];
      // 최대 로그 개수 제한으로 메모리 사용량 제어
      if (newLogs.length > maxLogs) {
        return newLogs.slice(-maxLogs);
      }
      
      // 첫 번째 로그가 추가된 후 자동 스크롤 비활성화
      if (autoScroll && prev.length === 0) {
        // 첫 번째 로그인 경우 자동 스크롤 유지
      } else if (autoScroll) {
        // 첫 번째 로그 이후 자동 스크롤 비활성화
        setTimeout(() => setAutoScroll(false), 100);
      }
      
      return newLogs;
    });

    // 기존 정규식 파싱 (하위 호환성)
    const roundMatch = message.match(/라운드\s*(\d+)(?:\/\d+)?\s*시작/);
    const accMatch = message.match(/🎯\s*학습 후 정확도:\s*([\d.]+)%/);
    if (accMatch) {
      const accuracy = parseFloat(accMatch[1]);
      
      let round = roundMatch
        ? parseInt(roundMatch[1], 10)
        : lastRoundRef.current;

      lastRoundRef.current = round;

      setAccuracyData((prev) => {
        // 중복 방지: 같은 라운드 데이터가 이미 있으면 갱신하지 않음
        if (prev.some((d) => d.round === round)) return prev;
        return [...prev, { round, accuracy }];
      });
    }

    const roundStartMatch = message.match(/=== 라운드 (\d+)\/\d+ 시작 ===/);
    if (roundStartMatch) {
      lastRoundRef.current = parseInt(roundStartMatch[1], 10);
    }
  };

  // 실시간 로그 스트리밍
  const startLogStreaming = () => {
    updateTrainingLog("📋 실시간 로그 스트리밍 시작...");

    // Server-Sent Events를 사용한 실시간 로그 스트리밍
    const eventSource = new EventSource("/api/fedhybrid/logs");

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.message) {
          // 로그 타입에 따른 처리
          if (data.type === "python_output") {
            // Python 출력 로그
            updateTrainingLog(`🐍 ${data.message}`);
            
            // 학습 완료 키워드 감지 (엑셀 파일 생성 완료 후에만)
            // "엑셀 파일 생성 완료" 메시지가 포함된 경우에만 학습 완료로 인식
            if (data.message.includes("엑셀 파일 생성 완료")) {
              setTrainingStatus("로컬 학습이 완료되었습니다!");
              // 학습 완료 이벤트 발생 (ExcelResultViewer 새로고침 트리거)
              const trainingCompleteEvent = new CustomEvent("training-complete", {
                detail: { timestamp: new Date().toISOString() },
              });
              window.dispatchEvent(trainingCompleteEvent);
            }
          } else if (data.type === "python_error") {
            // Python 에러 로그
            updateTrainingLog(`❌ ${data.message}`);
          } else if (data.type === "success") {
            // 학습 성공 메시지
            updateTrainingLog(data.message);
            setTrainingStatus("로컬 학습이 완료되었습니다!");
            // 학습 완료 이벤트 발생
            const trainingCompleteEvent = new CustomEvent("training-complete", {
              detail: { timestamp: new Date().toISOString() },
            });
            window.dispatchEvent(trainingCompleteEvent);
          } else if (data.type === "heartbeat") {
            // 하트비트는 표시하지 않음
            return;
          } else {
            // 일반 로그
            updateTrainingLog(data.message);
          }
        }
      } catch (error) {
        console.error("로그 파싱 오류:", error);
      }
    };

    eventSource.onerror = (error) => {
      console.error("로그 스트리밍 오류:", error);
      // 연결이 정상적으로 종료된 경우 (readyState === 2)는 오류가 아님
      if (eventSource.readyState === EventSource.CLOSED) {
        updateTrainingLog("🔄 실시간 로그 스트리밍 연결 종료");
        setIsLoading(false);
      } else {
        updateTrainingLog("❌ 로그 스트리밍 연결 오류");
        // 오류 발생 시에만 연결 종료
        eventSource.close();
        setIsLoading(false);
      }
    };

    return eventSource;
  };

  // 학습 시작
  const startTraining = async () => {
    if (!uploadedFile) {
      setError("파일을 선택해주세요.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setTrainingLogs([]);
    setShowLogs(true); // 로그를 항상 표시
    setAutoScroll(true); // 학습 시작 시에만 자동 스크롤 활성화
    setTrainingStatus("모델 다운로드 중...");

    // 로그 스트리밍 시작
    let eventSource: EventSource | null = null;

    try {
      // 1단계: 모델 다운로드
      updateTrainingLog("📥 서버에서 글로벌 모델 다운로드 중...");
      const modelResponse = await fetch("/api/fedhybrid?action=get_model");
      if (!modelResponse.ok) {
        throw new Error("모델을 다운로드할 수 없습니다.");
      }
      setTrainingStatus("모델 다운로드 완료. 로컬 학습 시작 중...");
      updateTrainingLog("✅ 글로벌 모델 다운로드 완료");

      // 2단계: 실시간 로그 스트리밍 시작
      updateTrainingLog("🔄 실시간 로그 스트리밍 시작...");
      eventSource = startLogStreaming();

      // 3단계: 로컬 학습 시작
      updateTrainingLog("🚀 FedHybrid-AI 클라이언트로 로컬 학습 시작...");
      updateTrainingLog("📁 파일 업로드 중...");
      const formData = new FormData();
      formData.append("file", uploadedFile);

      updateTrainingLog("⚙️ Python 스크립트 실행 중...");
      setTrainingStatus("로컬 학습 진행 중... (실시간 로그를 확인하세요)");

      const response = await fetch("/api/fedhybrid/local-training", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || "로컬 학습을 시작할 수 없습니다.");
      }

      const data = await response.json();

      if (data.success) {
        // 학습이 시작되었음을 표시 (완료가 아님!)
        setTrainingStatus("로컬 학습 진행 중... (실시간 로그를 확인하세요)");
        updateTrainingLog("✅ 학습 프로세스가 시작되었습니다 (PID: " + (data.processId || 'N/A') + ")");
        updateTrainingLog("📊 실시간 로그를 확인하여 학습 진행 상황을 모니터링하세요.");
        updateTrainingLog("⏳ 학습이 완료되면 예측 결과 파일이 생성됩니다.");
        
        // 학습 완료는 로그 스트리밍에서 확인 (Python 프로세스 종료 시)
        // 학습 완료 이벤트는 여기서 발생시키지 않음!
      } else {
        throw new Error(data.error || "학습을 시작할 수 없습니다.");
      }

      // 서버 상태 다시 확인 (학습 완료 후)
      // 학습이 완료될 때까지 기다리지 않고, 로그 스트리밍에서 완료를 감지
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다."
      );
      updateTrainingLog(
        `❌ 오류 발생: ${
          err instanceof Error ? err.message : "알 수 없는 오류"
        }`
      );
      setTrainingStatus("학습 중 오류가 발생했습니다.");
      
      // 오류 발생 시 로그 스트리밍 종료
      if (eventSource) {
        eventSource.close();
        updateTrainingLog("🔄 실시간 로그 스트리밍 종료");
      }
      setIsLoading(false);
    }
    // finally 블록 제거: 로그 스트리밍은 학습이 완료될 때까지 유지
  };

  // 예측 결과 다운로드
  const downloadPredictions = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/fedhybrid/local-training");

      if (!response.ok) {
        throw new Error("예측 결과를 다운로드할 수 없습니다.");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "prediction_results.xlsx";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다."
      );
    } finally {
      setIsLoading(false);
    }
  };

  // 모델 다운로드
  const downloadModel = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/fedhybrid?action=get_model");

      if (!response.ok) {
        throw new Error("모델을 다운로드할 수 없습니다.");
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "global_model.pth";
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다."
      );
    } finally {
      setIsLoading(false);
    }
  };

  // 컴포넌트 마운트 시 서버 상태 확인 (안전한 방식으로)
  useEffect(() => {
    const initializeComponent = async () => {
      try {
        await checkServerStatus();
      } catch (err) {
        console.error("초기화 중 오류 발생:", err);
        // 서버 상태 확인 실패해도 컴포넌트는 정상 렌더링
      }
    };

    initializeComponent();
  }, []);

  // 차트 커스텀 툴팁
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <TooltipBox>
          <TooltipLabel>{`Round ${label}`}</TooltipLabel>
          <TooltipValue>
            <span>정확도: </span>
            {`${payload[0].value}%`}
          </TooltipValue>
        </TooltipBox>
      );
    }
    return null;
  };

  // 로딩 중에도 전체 컴포넌트를 표시하되, 버튼만 비활성화

  return (
    <Wrapper className={className}>
      <Card>
        <Title>FedHybrid-AI 연동</Title>

        {error && (
          <ErrorWrapper>
            <ErrorTitle>오류 발생</ErrorTitle>
            <ErrorDescription>{error}</ErrorDescription>
          </ErrorWrapper>
        )}

        {trainingStatus && (
          <StatusWrapper>
            <StatusTitle>학습 진행 상황</StatusTitle>
            <StatusMessage>{trainingStatus}</StatusMessage>
          </StatusWrapper>
        )}

        {/* 서버 상태 */}
        <Section>
          <SectionTitle>서버 상태</SectionTitle>
          {serverStatus ? (
            <StatusGrid>
              <StatusCard>
                <StatusLabel>현재 라운드</StatusLabel>
                <StatusValue>{serverStatus.current_round}</StatusValue>
              </StatusCard>
              <StatusCard>
                <StatusLabel>활성 라운드</StatusLabel>
                <StatusValue>
                  {Object.keys(serverStatus.active_rounds).length}
                </StatusValue>
              </StatusCard>
              <StatusCard>
                <StatusLabel>목표 클라이언트</StatusLabel>
                <StatusValue>
                  {serverStatus.round_config?.target_clients || "N/A"}
                </StatusValue>
              </StatusCard>
            </StatusGrid>
          ) : (
            <NoDataMessage>서버에 연결할 수 없습니다.</NoDataMessage>
          )}

          <Button onClick={checkServerStatus} disabled={isLoading}>
            상태 새로고침
          </Button>
        </Section>

        {/* 파일 업로드 및 로컬 학습 */}
        <Section>
          <SectionTitle>로컬 학습</SectionTitle>
          <TrainingDescription>
            <p>
              1. CSV 또는 Excel 파일을 업로드하면 서버에서 모델을
              다운로드합니다.
            </p>
            <p>2. Excel 파일은 자동으로 CSV로 변환됩니다.</p>
            <p>3. 다운로드한 모델로 로컬에서 학습을 진행합니다.</p>
            <p>4. 학습 완료 후 예측 결과를 생성합니다.</p>
            <p>
              <strong>
                ⚠️ 주의: 예측 결과 파일(diabetic_predictions.xlsx,
                prediction_results.xlsx)은 업로드할 수 없습니다.
              </strong>
            </p>
          </TrainingDescription>
          <UploadWrapper>
            <FileInput
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={handleFileUpload}
            />
            {uploadedFile && (
              <FileInfo>선택된 파일: {uploadedFile.name}</FileInfo>
            )}
            <Button
              onClick={startTraining}
              disabled={!uploadedFile || isLoading}
              fullWidth
            >
              {isLoading ? (
                <>
                  <Spinner
                    style={{
                      width: "1rem",
                      height: "1rem",
                      marginRight: "0.5rem",
                    }}
                  />
                  학습 진행 중...
                </>
              ) : (
                "모델 다운로드 & 로컬 학습 시작"
              )}
            </Button>
          </UploadWrapper>
        </Section>

        {/* 결과 다운로드 */}
        <Section>
          <SectionTitle>결과 다운로드</SectionTitle>
          <DownloadGrid>
            <Button
              onClick={downloadPredictions}
              disabled={isLoading}
              variant="outline"
            >
              예측 결과 다운로드
            </Button>
            <Button
              onClick={downloadModel}
              disabled={isLoading}
              variant="outline"
            >
              모델 다운로드
            </Button>
          </DownloadGrid>
        </Section>

        {/* 정확도 차트 */}
        {trainingLogs.length > 0 && (
          <Section>
            <Header>
              <SectionTitle>Round별 정확도 차트</SectionTitle>
              <LegendInfo>
                <LegendItem>
                  <LegendDot />
                  <span>정확도 (%)</span>
                </LegendItem>
                <span>총 {accuracyData.length} 라운드</span>
              </LegendInfo>
            </Header>

            <ChartWrapper height={400}>
              <ResponsiveContainer>
                <LineChart
                  data={accuracyData}
                  margin={{
                    top: 30,
                    right: 30,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis
                    dataKey="round"
                    stroke="#6b7280"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    stroke="#6b7280"
                    fontSize={12}
                    tickLine={false}
                    axisLine={false}
                    domain={[0, 100]}
                    tickFormatter={(value) => `${value}%`}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="accuracy"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ fill: "#3b82f6", strokeWidth: 2, r: 4 }}
                    activeDot={{
                      r: 6,
                      stroke: "#3b82f6",
                      strokeWidth: 2,
                      fill: "#ffffff",
                    }}
                    name="정확도 (%)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartWrapper>

            <Stats>
              <StatItem>
                <StatLabel>최고 정확도</StatLabel>
                <StatValue color="#16a34a">
                  {accuracyData.length > 0
                    ? Math.max(...accuracyData.map((d) => d.accuracy)).toFixed(2)
                    : "0.00"}%
                </StatValue>
              </StatItem>
              <StatItem>
                <StatLabel>최저 정확도</StatLabel>
                <StatValue color="#dc2626">
                  {accuracyData.length > 0
                    ? Math.min(...accuracyData.map((d) => d.accuracy)).toFixed(2)
                    : "0.00"}%
                </StatValue>
              </StatItem>
              <StatItem>
                <StatLabel>평균 정확도</StatLabel>
                <StatValue color="#2563eb">
                  {accuracyData.length > 0
                    ? (
                        accuracyData.reduce((sum, d) => sum + d.accuracy, 0) /
                        accuracyData.length
                      ).toFixed(2)
                    : "0.00"}
                  %
                </StatValue>
              </StatItem>
              <StatItem>
                <StatLabel>현재 라운드</StatLabel>
                <StatValue>{accuracyData.length}</StatValue>
              </StatItem>
            </Stats>
          </Section>
        )}

        {/* 학습 로그 */}
        {trainingLogs.length > 0 && (
          <Section>
            <SectionTitle>학습 로그</SectionTitle>
            <LogWrapper>
              <LogHeader>
                <LogTitle>실시간 학습 과정</LogTitle>
                <LogToggle onClick={() => setShowLogs(!showLogs)}>
                  {showLogs ? "로그 숨기기" : "로그 보기"}
                </LogToggle>
              </LogHeader>
              {showLogs && (
                <LogContent>
                  {trainingLogs.length === 0 ? (
                    <LogLine style={{ color: "#6b7280", fontStyle: "italic" }}>
                      아직 로그가 없습니다. 학습을 시작하면 로그가 표시됩니다.
                    </LogLine>
                  ) : (
                    <div>
                      {trainingLogs.map((log, index) => (
                        <LogLine
                          key={`log-${index}-${log.slice(0, 20)}`}
                          style={{
                            color: log.includes("❌")
                              ? "#ef4444"
                              : log.includes("✅")
                              ? "#10b981"
                              : log.includes("🔄")
                              ? "#3b82f6"
                              : log.includes("🐍")
                              ? "#10b981"
                              : "#d1d5db",
                          }}
                        >
                          {log}
                        </LogLine>
                      ))}
                      {/* 자동 스크롤을 위한 빈 div */}
                      <div
                        ref={(el) => {
                          if (el && autoScroll) {
                            el.scrollIntoView({ behavior: "smooth" });
                          }
                        }}
                      />
                    </div>
                  )}
                </LogContent>
              )}
            </LogWrapper>
          </Section>
        )}

        {/* 성능 지표 */}
        {serverStatus?.global_accuracies &&
          serverStatus.global_accuracies.length > 0 && (
            <Section>
              <SectionTitle>학습 성능</SectionTitle>
              <PerformanceWrapper>
                <PerformanceLabel>최근 정확도</PerformanceLabel>
                <PerformanceList>
                  {serverStatus.global_accuracies
                    .slice(-5)
                    .map((accuracy, index) => (
                      <PerformanceItem key={index}>
                        <PerformanceRound>
                          라운드{" "}
                          {serverStatus.global_accuracies.length - 4 + index}
                        </PerformanceRound>
                        <PerformanceValue>
                          {(accuracy * 100).toFixed(2)}%
                        </PerformanceValue>
                      </PerformanceItem>
                    ))}
                </PerformanceList>
              </PerformanceWrapper>
            </Section>
          )}
      </Card>
    </Wrapper>
  );
}

// Styled Components
const Wrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1.5rem;
`;

const Card = styled.div`
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  padding: 2rem;
`;

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: 700;
  color: #111827;
  margin-bottom: 1.5rem;
`;

const Section = styled.div`
  margin-bottom: 2rem;
`;

const SectionTitle = styled.h3`
  font-size: 1.125rem;
  font-weight: 600;
  color: #374151;
  margin-bottom: 1rem;
`;

const StatusGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
  margin-bottom: 1rem;
`;

const StatusCard = styled.div`
  background-color: #f9fafb;
  padding: 1rem;
  border-radius: 0.5rem;
  text-align: center;
`;

const StatusLabel = styled.p`
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 0.5rem;
`;

const StatusValue = styled.p`
  font-size: 1.5rem;
  font-weight: 700;
  color: #111827;
`;

const Button = styled.button<{ variant?: string; fullWidth?: boolean }>`
  background-color: ${(props) =>
    props.variant === "outline" ? "transparent" : "#2563eb"};
  color: ${(props) => (props.variant === "outline" ? "#2563eb" : "white")};
  border: 2px solid #2563eb;
  padding: 0.75rem 1.5rem;
  border-radius: 0.5rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  width: ${(props) => (props.fullWidth ? "100%" : "auto")};

  &:hover:not(:disabled) {
    background-color: ${(props) =>
      props.variant === "outline" ? "#2563eb" : "#1d4ed8"};
    color: ${(props) => (props.variant === "outline" ? "white" : "white")};
  }

  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const UploadWrapper = styled.div`
  display: flex;
  flex-direction: column;
  gap: 1rem;
`;

const FileInput = styled.input`
  display: block;
  width: 100%;
  padding: 0.75rem;
  border: 2px dashed #d1d5db;
  border-radius: 0.5rem;
  background-color: #f9fafb;
  cursor: pointer;

  &:hover {
    border-color: #2563eb;
    background-color: #eff6ff;
  }
`;

const FileInfo = styled.p`
  font-size: 0.875rem;
  color: #6b7280;
`;

const DownloadGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 1rem;
`;

const PerformanceWrapper = styled.div`
  background-color: #f9fafb;
  padding: 1.5rem;
  border-radius: 0.5rem;
`;

const PerformanceLabel = styled.p`
  font-size: 0.875rem;
  color: #6b7280;
  margin-bottom: 1rem;
`;

const PerformanceList = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
`;

const PerformanceItem = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const PerformanceRound = styled.span`
  font-size: 0.875rem;
  color: #374151;
`;

const PerformanceValue = styled.span`
  font-weight: 600;
  color: #111827;
`;

const ErrorWrapper = styled.div`
  background-color: #fef2f2;
  border: 1px solid #fecaca;
  border-radius: 0.5rem;
  padding: 1rem;
  margin-bottom: 1.5rem;
`;

const ErrorTitle = styled.h3`
  font-size: 1rem;
  font-weight: 600;
  color: #dc2626;
  margin-bottom: 0.5rem;
`;

const ErrorDescription = styled.p`
  color: #7f1d1d;
  font-size: 0.875rem;
`;

const StatusWrapper = styled.div`
  background-color: #eff6ff;
  border: 1px solid #bfdbfe;
  border-radius: 0.5rem;
  padding: 1rem;
  margin-bottom: 1.5rem;
`;

const StatusMessage = styled.p`
  color: #1e40af;
  font-size: 0.875rem;
`;

const StatusTitle = styled.h4`
  color: #1e40af;
  font-size: 1rem;
  font-weight: 600;
  margin-bottom: 0.5rem;
`;

const NoDataMessage = styled.p`
  color: #6b7280;
  font-size: 0.875rem;
  margin-bottom: 1rem;
`;

const LoadingWrapper = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
`;

const LoadingContent = styled.div`
  text-align: center;
`;

const Spinner = styled.div`
  width: 3rem;
  height: 3rem;
  border: 3px solid #e5e7eb;
  border-top: 3px solid #2563eb;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem auto;

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
`;

const LoadingMessage = styled.p`
  color: #6b7280;
  font-size: 0.875rem;
`;

const TrainingDescription = styled.div`
  background-color: #f0f9ff;
  border: 1px solid #bae6fd;
  border-radius: 0.5rem;
  padding: 1rem;
  margin-bottom: 1rem;

  p {
    margin: 0.25rem 0;
    color: #0369a1;
    font-size: 0.875rem;
  }
`;

const LogWrapper = styled.div`
  background-color: #1f2937;
  border-radius: 0.5rem;
  overflow: hidden;
`;

const LogHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem;
  background-color: #374151;
  border-bottom: 1px solid #4b5563;
`;

const LogTitle = styled.h4`
  color: #f9fafb;
  font-size: 1rem;
  font-weight: 600;
  margin: 0;
`;

const LogToggle = styled.button`
  background-color: #3b82f6;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 0.25rem;
  font-size: 0.875rem;
  cursor: pointer;
  transition: background-color 0.2s;

  &:hover {
    background-color: #2563eb;
  }
`;

const LogContent = styled.div`
  max-height: 400px;
  overflow-y: auto;
  padding: 1rem;
  background-color: #111827;
`;

const LogLine = styled.div`
  color: #d1d5db;
  font-family: "Courier New", monospace;
  font-size: 0.875rem;
  line-height: 1.4;
  margin-bottom: 0.25rem;
  word-break: break-all;
  white-space: pre-wrap;

  /* Python 출력 로그 스타일 */
  &:contains("🐍") {
    color: #10b981;
  }

  /* 에러 로그 스타일 */
  &:contains("❌") {
    color: #ef4444;
  }

  /* 성공 로그 스타일 */
  &:contains("✅") {
    color: #10b981;
  }

  /* 진행 중 로그 스타일 */
  &:contains("🔄") {
    color: #3b82f6;
  }
`;

// 정확도 차트 스타일
const Header = styled.div`
  margin-bottom: 1.5rem;
`;

const LegendInfo = styled.div`
  display: flex;
  align-items: center;
  gap: 1rem;
  font-size: 0.875rem;
  color: #4b5563;
`;

const LegendItem = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const LegendDot = styled.div`
  width: 0.75rem;
  height: 0.75rem;
  background-color: #3b82f6;
  border-radius: 50%;
`;

const ChartWrapper = styled.div<{ height: number }>`
  width: 100%;
  height: ${({ height }) => height}px;
`;

const Stats = styled.div`
  margin-top: 1.5rem;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
  padding-top: 1rem;
  border-top: 1px solid #f3f4f6;

  @media (min-width: 768px) {
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }
`;

const StatItem = styled.div`
  text-align: center;
`;

const StatLabel = styled.p`
  font-size: 0.875rem;
  color: #4b5563;
`;

const StatValue = styled.p<{ color?: string }>`
  font-size: 1.125rem;
  font-weight: 600;
  color: ${({ color }) => color || "#111827"};
`;

const TooltipBox = styled.div`
  background: white;
  padding: 0.75rem;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
`;

const TooltipLabel = styled.p`
  font-size: 0.875rem;
  font-weight: 500;
  color: #111827;
`;

const TooltipValue = styled.p`
  font-size: 0.875rem;
  color: #2563eb;

  span {
    font-weight: 500;
  }
`;
