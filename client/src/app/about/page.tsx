"use client";

import styled from "styled-components";
import { useRouter } from "next/navigation";
import { Wifi, Target, Scale } from "lucide-react";
import FeatureCard from "@/components/common/Card";
import { featureTags } from "@/constants/FeatureTags";
import { Path } from "@/constants/Path";

export default function About() {
  const router = useRouter();

  return (
    <PageWrapper>
      <Section>
        <CenteredContainer>
          <LargeHeading>
            Introducing Federated Learning <Hy>Hy</Hy>
            <Brid>Brid</Brid>
          </LargeHeading>

          <SubHeading>: A Novel Approach to Federated Learning</SubHeading>

          <TagLine>{featureTags.map((tag) => tag.feature)}</TagLine>

          {/* <Description>차세대 연합 학습 알고리즘, FedHB를 소개합니다.</Description> */}
        </CenteredContainer>
      </Section>

      <Section bg="#f9fafb">
        <div style={{ maxWidth: "80rem", margin: "0 auto" }}>
          <Grid>
            <FeatureCard
              icon={<Wifi size={48} color="#3b82f6" />}
              title="통신 비용 감소"
              description="효율적인 모델 압축과 선택적 업데이트를 통해 네트워크 통신 비용을 대폭 줄입니다."
            />

            <FeatureCard
              icon={<Target size={48} color="#22c55e" />}
              title="정확도 개선"
              description="하이브리드 학습 방식을 통해 기존 연합학습 대비 높은 모델 정확도를 달성합니다."
            />

            <FeatureCard
              icon={<Scale size={48} color="#8b5cf6" />}
              title="공정성 향상"
              description="모든 참여자에게 공평한 학습 기회를 제공하여 편향 없는 모델을 구축합니다."
            />
          </Grid>
        </div>
      </Section>

      <ButtonSection>
        <ButtonWrapper>
          <StyledButton onClick={() => router.push(Path.DASHBOARD)}>
            FedHB 성능 바로 확인하기
          </StyledButton>
        </ButtonWrapper>
      </ButtonSection>
    </PageWrapper>
  );
}

const PageWrapper = styled.div`
  background-color: white;
  height: 100%;
`;

const Section = styled.section<{ bg?: string }>`
  padding: 1rem 1.5rem;
`;

const ButtonSection = styled.section<{ bg?: string }>`
  padding: 3rem 1.5rem;
`;

const CenteredContainer = styled.div`
  max-width: 64rem;
  margin: 0 auto;
  text-align: center;
`;

const LargeHeading = styled.h1`
  font-size: 3rem;
  font-weight: bold;
  color: #111827;
  margin-bottom: 1rem;
  text-shadow: 0 1px 1px rgba(0, 0, 0, 0.05);

  @media (min-width: 768px) {
    font-size: 3.75rem;
  }
`;

const Hy = styled.span`
  color: #7e22ce;
`;

const Brid = styled.span`
  color: #3b82f6;
`;

const SubHeading = styled.p`
  font-size: 1.125rem;
  color: #6b7280;
  margin-bottom: 2rem;

  @media (min-width: 768px) {
    font-size: 1.25rem;
  }
`;

const TagLine = styled.p`
  font-size: 1.25rem;
  font-weight: bold;
  color: #111827;

  @media (min-width: 768px) {
    font-size: 1.5rem;
  }
`;

const Description = styled.p`
  font-size: 1.125rem;
  color: #374151;
  margin-bottom: 4rem;
`;

const Grid = styled.div`
  display: grid;
  gap: 2rem;

  @media (min-width: 768px) {
    grid-template-columns: repeat(3, 1fr);
  }
`;

const ButtonWrapper = styled.div`
  max-width: 64rem;
  margin: 0 auto;
  text-align: center;
`;

const StyledButton = styled.button`
  color: #1e3a8a;
  font-weight: bold;
  font-size: 1.25rem;
  padding: 1rem 3rem;
  background-color: #dbeafe;
  cursor: pointer;
  border-radius: 0.5rem;
  border: none;
  transition: all 0.2s ease-in-out;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);

  &:hover {
    background-color: #bfdbfe;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }
`;
