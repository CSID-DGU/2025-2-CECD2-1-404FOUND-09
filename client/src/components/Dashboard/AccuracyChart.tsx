"use client";

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
import React from "react";

interface AccuracyChartProps {
  title?: string;
  height?: number;
}

interface AccuracyPoint {
  round: number;
  accuracy: number;
}

// TODO: 더미데이터 삭제
const accuracyData: AccuracyPoint[] = Array.from({ length: 50 }, (_, i) => ({
  round: i + 1,
  accuracy: Number((45 + Math.random() * 25).toFixed(2)),
}));

export function AccuracyChart({
  title = "Round별 정확도",
  height = 400,
}: AccuracyChartProps) {
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

  return (
    <Card>
      <Header>
        <Title>{title}</Title>
        <LegendInfo>
          <LegendItem>
            <LegendDot />
            <span>정확도 (%)</span>
          </LegendItem>
          <span>총 {accuracyData.length} 라운드</span>
        </LegendInfo>
      </Header>

      <ChartWrapper height={height}>
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
            {Math.max(...accuracyData.map((d) => d.accuracy)).toFixed(2)}%
          </StatValue>
        </StatItem>
        <StatItem>
          <StatLabel>최저 정확도</StatLabel>
          <StatValue color="#dc2626">
            {Math.min(...accuracyData.map((d) => d.accuracy)).toFixed(2)}%
          </StatValue>
        </StatItem>
        <StatItem>
          <StatLabel>평균 정확도</StatLabel>
          <StatValue color="#2563eb">
            {(
              accuracyData.reduce((sum, d) => sum + d.accuracy, 0) /
              accuracyData.length
            ).toFixed(2)}
            %
          </StatValue>
        </StatItem>
        <StatItem>
          <StatLabel>현재 라운드</StatLabel>
          <StatValue>{accuracyData.length}</StatValue>
        </StatItem>
      </Stats>
    </Card>
  );
}

const Card = styled.div`
  background: #ffffff;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  padding: 1.5rem;
`;

const Header = styled.div`
  margin-bottom: 1.5rem;
`;

const Title = styled.h3`
  font-size: 1.25rem;
  font-weight: 600;
  color: #111827;
  margin-bottom: 0.5rem;
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
