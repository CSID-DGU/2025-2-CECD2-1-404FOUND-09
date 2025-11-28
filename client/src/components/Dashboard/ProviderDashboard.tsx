"use client";

import DashboardCard from "./DashboardCard";
import FederationView from "../federations/FederationView";
import styled from "styled-components";
import ExcelResultViewer from "./ExcelResultViewer";
import { AccuracyChart } from "./AccuracyChart";

export default function ProviderDashboard() {
  return (
    <Wrapper>
      <Title>관리자 대시보드</Title>
      <DashboardCard />
      <FederationView />
      <AccuracyChart />
      <ExcelResultViewer />
    </Wrapper>
  );
}

const Wrapper = styled.div``;

const Title = styled.h1`
  font-size: 2.25rem;
  font-weight: 700;
  color: #111827;
  text-align: center;
`;
