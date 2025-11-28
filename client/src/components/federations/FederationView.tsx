"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import LoadingView from "../common/LoadingView";
import ErrorView from "../common/ErrorView";
import FederationCard from "./FederationCard";
import styled from "styled-components";
import { Users } from "lucide-react";
import { Path } from "@/constants/Path";

export default function FederationView() {
  const [loading, setLoading] = useState(true);
  const [federation, setFederation] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();

  useEffect(() => {
    const fetchFederation = async () => {
      setLoading(true);
      try {
        const res = await fetch("/api/federation");
        if (res.status == 204) {
          // 204 -> 연합정보 없음
          setFederation(null);
        } else {
          const data = await res.json();
          if (!res.ok) throw new Error(data.error);
          setFederation(data);
        }
      } catch (e: any) {
        setError(e.message || "조회 실패");
      }
      setLoading(false);
    };
    fetchFederation();
  }, []);

  if (loading) {
    return <LoadingView />;
  }

  if (error) {
    return <ErrorView message={error} />;
  }

  if (!federation) {
    return (
      <EmptyFederationCard
        onClick={() => router.push(Path.FEDERATION_UPDATE)}
      />
    );
  }

  return (
    <FederationCard
      federation={{
        id: federation.id,
        name: federation.name,
        instance_id: federation.instance_id ?? "(없음)",
      }}
    />
  );
}

type EmptyFederationCardProps = {
  onClick?: () => void;
};

function EmptyFederationCard({ onClick }: EmptyFederationCardProps) {
  return (
    <Card onClick={onClick}>
      <Content>
        <IconWrapper>
          <Users className="icon" />
        </IconWrapper>
        <TextGroup>
          <Title>연합 없음</Title>
          <Description>
            아직 생성된 연합이 없습니다. 클릭하여 새로운 연합을 생성해보세요!
          </Description>
        </TextGroup>
      </Content>
    </Card>
  );
}

const Card = styled.div`
  padding: 2rem 0;
  background-color: #f9fafb;
  border: 2px dashed #d1d5db;
  border-radius: 0.5rem;
  text-align: center;
`;

const Content = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1rem;
`;

const IconWrapper = styled.div`
  width: 4rem;
  height: 4rem;
  background-color: #e5e7eb;
  border-radius: 9999px;
  display: flex;
  align-items: center;
  justify-content: center;

  .icon {
    width: 2rem;
    height: 2rem;
    color: #9ca3af;
  }
`;

const TextGroup = styled.div``;

const Title = styled.h3`
  font-size: 1.125rem;
  font-weight: 500;
  color: #111827;
  margin-bottom: 0.5rem;
`;

const Description = styled.p`
  color: #4b5563;
`;
