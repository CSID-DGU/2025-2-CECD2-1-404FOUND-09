"use client";

import { BarChart3, List } from "lucide-react";
import MainButton from "@/components/common/Button";
import FederatedLearningDiagram from "@/components/home/FederatedLearningDiagram";
import { useRouter } from "next/navigation";
import styled from "styled-components";
import { Path } from "@/constants/Path";

export default function Home() {
  const router = useRouter();

  return (
    <Container>
      <MainTitleWrapper>
        <MainTitle>
          Federated Learning - <Hy>Hy</Hy>
          <Brid>Brid</Brid> Demonstration
        </MainTitle>
      </MainTitleWrapper>

      <DiagramWrapper>
        <DiagramBox>
          <FederatedLearningDiagram />
        </DiagramBox>
      </DiagramWrapper>

      <ButtonGrid>
        <MainButton
          icon={<BarChart3 size={40} color="#1e3a8a" />}
          label="대시보드"
          onClick={() => router.push(Path.DASHBOARD)}
        />
        <MainButton
          icon={<List size={40} color="#1e3a8a" />}
          label="인스턴스"
          onClick={() => router.push(Path.INSTANCE)}
        />
      </ButtonGrid>
    </Container>
  );
}

const Container = styled.div`
  max-width: 72rem;
  height: 100%;
  margin: 0 auto;
  padding: 0;
`;

const MainTitleWrapper = styled.div`
  text-align: center;
  padding-top: 1rem;
  margin-bottom: 4rem;
`;

const MainTitle = styled.h1`
  font-size: 3rem;

  @media (min-width: 768px) {
    font-size: 3.75rem;
  }

  font-weight: 700;
  color: #111827;
  margin-bottom: 1rem;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
`;

const Hy = styled.span`
  color: #9333ea;
`;

const Brid = styled.span`
  color: #2563eb;
`;

const DiagramWrapper = styled.div`
  display: flex;
  justify-content: center;
  margin-bottom: 4rem;
`;

const DiagramBox = styled.div`
  background-color: white;
  padding: 2rem;
  border-radius: 0.5rem;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  border: 1px solid #e5e7eb;
`;

const ButtonGrid = styled.div`
  display: grid;
  grid-template-columns: 1fr;
  gap: 2rem;
  padding: 1rem;
  max-width: 64rem;
  margin: 0 auto;

  @media (min-width: 768px) {
    grid-template-columns: repeat(2, 1fr);
  }
`;
