'use client'

import styled, { keyframes } from 'styled-components'

export default function LoadingView() {
  return (
    <Wrapper>
      <Content>
        <Spinner />
        <Message>데이터를 불러오는 중...</Message>
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

const spin = keyframes`
  to {
    transform: rotate(360deg);
  }
`

const Spinner = styled.div`
  margin: 0 auto 1rem auto;
  height: 3rem;
  width: 3rem;
  border-radius: 9999px;
  border-bottom: 2px solid #2563eb;
  animation: ${spin} 1s linear infinite;
`

const Message = styled.p`
  color: #4b5563;
`
