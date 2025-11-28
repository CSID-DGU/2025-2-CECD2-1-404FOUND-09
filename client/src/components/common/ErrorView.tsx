'use client'

import styled from 'styled-components'

interface ErrorViewProps {
  message: string
}

export default function ErrorView({ message }: ErrorViewProps) {
  return (
    <Wrapper>
      <Content>
        <Title>오류 발생</Title>
        <Description>{message}</Description>
      </Content>
    </Wrapper>
  )
}

const Wrapper = styled.div`
  background-color: white;
  height: 90%;
  display: flex;
  align-items: center;
  justify-content: center;
`

const Content = styled.div`
  text-align: center;
`

const Title = styled.h2`
  font-size: 1.5rem;
  font-weight: 700;
  color: #dc2626;
  margin-bottom: 1rem;
`

const Description = styled.p`
  color: #4b5563;
`
