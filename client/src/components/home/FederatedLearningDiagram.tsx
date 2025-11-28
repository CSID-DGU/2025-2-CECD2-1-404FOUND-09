'use client'

import styled from 'styled-components'

export default function FederatedLearningDiagram() {
  return (
    <DiagramContainer>
      <ServerContainer>
        <ServerBox>
          <ServerInner />
        </ServerBox>
        <ServerLabel>SERVER</ServerLabel>
      </ServerContainer>

      <ArrowContainer>
        <ArrowLine />
        <ArrowHead />
        <ArrowLine />
      </ArrowContainer>

      <NodeGrid>
        {[1, 2, 3, 4].map((node) => (
          <NodeWrapper key={node}>
            <NodeBox>
              <NodeInner />
            </NodeBox>
            <NodeLabel>NODE {node}</NodeLabel>
          </NodeWrapper>
        ))}
      </NodeGrid>
    </DiagramContainer>
  )
}

const DiagramContainer = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 2rem;
`

const ServerContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`

const ServerBox = styled.div`
  width: 4rem;
  height: 4rem;
  background-color: #9333ea;
  border-radius: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 0.5rem;
`

const ServerInner = styled.div`
  width: 2rem;
  height: 2rem;
  background-color: white;
  border-radius: 0.25rem;
`

const ServerLabel = styled.span`
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
`

const ArrowContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
`

const ArrowLine = styled.div`
  width: 2rem;
  height: 2px;
  background-color: #9ca3af;
`

const ArrowHead = styled.div`
  width: 0;
  height: 0;
  border-bottom: 4px solid transparent;
  border-top: 4px solid transparent;
  border-left: 6px solid #9ca3af;
`

const NodeGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 1rem;
`

const NodeWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
`

const NodeBox = styled.div`
  width: 3rem;
  height: 3rem;
  background-color: #3b82f6;
  border-radius: 0.5rem;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 0.25rem;
`

const NodeInner = styled.div`
  width: 1.5rem;
  height: 1.5rem;
  background-color: white;
  border-radius: 0.125rem;
`

const NodeLabel = styled.span`
  font-size: 0.75rem;
  color: #6b7280;
`
