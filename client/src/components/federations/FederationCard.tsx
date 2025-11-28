"use client";

import { Server, Edit } from "lucide-react";
import styled from "styled-components";
import { useRouter } from "next/navigation";
import FederationButton from "../common/FederationButton";

interface Federation {
  id: string;
  name: string;
  instance_id?: string;
}

interface Props {
  federation: Federation;
}

export default function FederationCard({ federation }: Props) {
  const router = useRouter();

  return (
    <div>
      <Header>
        <div>
          <Title>내 연합</Title>
          <Name>{federation.name}</Name>
          <Grid>
            <InfoGroup>
              <InfoRow>
                <Server size={20} color="#6b7280" />
                <Label>Instance ID: {federation.instance_id ?? "(없음)"}</Label>
              </InfoRow>
            </InfoGroup>
          </Grid>
        </div>

        <FederationButton
          icon={<Edit />}
          label="연합 정보 수정"
          onClick={() =>
            router.push(
              `/supabase/federation/update?name=${encodeURIComponent(
                federation.name
              )}&instance_id=${encodeURIComponent(
                federation.instance_id ?? ""
              )}`
            )
          }
        />
      </Header>
    </div>
  );
}

const Card = styled.div`
  background-color: white;
  border-radius: 0.5rem;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  border: 1px solid #e5e7eb;
  padding: 1.5rem;
  max-width: 40rem;
  margin: 2rem auto;
`;

const Header = styled.div`
  display: flex;
  justify-content: space-between;
  padding-bottom: 0.5rem;
`;

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 0.5rem;
`;

const Name = styled.h3`
  font-size: 1.5rem;
  font-weight: 700;
  color: #2563eb;
  padding-bottom: 0.5rem;
`;

const StatusBadge = styled.div`
  background-color: #d1fae5;
  color: #065f46;
  padding: 0.25rem 0.75rem;
  border-radius: 9999px;
  font-size: 0.875rem;
  font-weight: 500;
  height: fit-content;
`;

const Grid = styled.div`
  display: grid;
  gap: 1.5rem;

  @media (min-width: 768px) {
    grid-template-columns: 1fr 1fr;
  }
`;

const InfoGroup = styled.div`
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
`;

const InfoRow = styled.div`
  display: flex;
  align-items: center;
  gap: 0.5rem;
`;

const Label = styled.span`
  font-size: 0.875rem;
  color: #6b7280;
`;

const ButtonWrapper = styled.button`
  padding-top: 1rem;
  border: none;
  background-color: white;
`;
