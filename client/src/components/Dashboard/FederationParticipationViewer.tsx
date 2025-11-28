"use client";

import React, { useEffect, useState } from "react";
import LoadingView from "../common/LoadingView";
import ErrorView from "../common/ErrorView";
import styled from "styled-components";
import { Hash, Copy } from "lucide-react";

export function FederationParticipationViewer() {
  const [loading, setLoading] = useState(true);
  const [participation, setParticipation] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchParticipation = async () => {
      setLoading(true);
      try {
        const res = await fetch("/api/participations");
        if (res.status == 204) {
          // 204 -> 정보 없음
          setParticipation(null);
        } else {
          const data = await res.json();
          if (!res.ok) throw new Error(data.error);
          setParticipation(data);
        }
      } catch (e: any) {
        setError(e.message || "조회 실패");
      }
      setLoading(false);
    };
    fetchParticipation();
  }, []);

  if (loading) {
    return <LoadingView />;
  }

  if (error) {
    return <ErrorView message={error} />;
  }

  if (!participation) {
    return (
      <Card>
        <Title>내 참여 연합</Title>
        <EmptyState>
          <Hash size={48} color="#9ca3af" />
          <p>참여 중인 연합이 없습니다.</p>
        </EmptyState>
      </Card>
    );
  }

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      alert("복사되었습니다!");
    } catch (err) {
      console.error("복사 실패:", err);
    }
  };

  return (
    <Card>
      <Header>
        <Title>내 참여 연합</Title>
      </Header>

      <Content>
        <Field>
          <FieldInfo>
            <Label>Federation ID</Label>
            <Code>{participation.federation_id}</Code>
          </FieldInfo>
          <CopyButton
            onClick={() => copyToClipboard(participation.federation_id)}
            title="복사"
          >
            <Copy size={16} />
          </CopyButton>
        </Field>

        <InfoGrid>
          <InfoRow>
            <Label>참여일:</Label>
            <Value>
              {new Date(participation.created_at).toLocaleDateString("ko-KR")}
            </Value>
          </InfoRow>

          <InfoRow>
            <Label>인스턴스:</Label>
            <Value>{participation.instance_id || "미할당"}</Value>
          </InfoRow>
        </InfoGrid>
      </Content>
    </Card>
  );
}

const Card = styled.div`
  background: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  padding: 1.5rem;
  margin-bottom: 2rem;
`;

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: 700;
  color: #111827;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  margin-bottom: 1rem;
`;

const Content = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
`;

const Field = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #f9fafb;
  border-radius: 0.5rem;
  padding: 0.75rem;
`;

const FieldInfo = styled.div`
  flex: 1;
  min-width: 0;
`;

const Label = styled.p`
  font-size: 0.875rem;
  color: #6b7280;
`;

const Code = styled.code`
  font-size: 0.875rem;
  font-family: monospace;
  color: #1f2937;
  word-break: break-all;
`;

const CopyButton = styled.button`
  margin-left: 0.75rem;
  border: none;
  padding: 0.5rem;
  background: #f9fafb;
  color: #6b7280;
  transition: color 0.2s;
  &:hover {
    color: #374151;
  }
`;

const InfoGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  padding-top: 0.3rem;
  font-size: 0.875rem;
`;

const InfoRow = styled.div`
  display: flex;
  align-items: center;
`;

const Value = styled.span`
  margin-left: 0.5rem;
  color: #111827;
`;

const EmptyState = styled.div`
  text-align: center;
  padding: 1.5rem 0;
  color: #6b7280;
`;
