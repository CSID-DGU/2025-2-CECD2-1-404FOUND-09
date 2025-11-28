'use client'

import styled from "styled-components";
import FederationList from "@/components/federations/FederationList";

export default function Instance() {
  return (
    <Container>
      <Content>
        <Title>인스턴스 리스트 조회</Title>
        <FederationList />
      </Content>
    </Container>
  )
}

const Container = styled.div`
  width: 100%;
  margin: 0 auto;
  padding: 2rem 2rem;
  background-color: white;
  `

const Content = styled.div`
  width: 100%;
  margin: 0 auto;
`

const Title = styled.h1`
  font-size: 2.25rem;
  font-weight: 700;
  color: #111827;
  text-align: center;
  margin-bottom: 3rem;
`