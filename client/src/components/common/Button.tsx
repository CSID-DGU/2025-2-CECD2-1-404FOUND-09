'use client'

import styled from 'styled-components'
import { ReactNode } from 'react'

type MainButtonProps = {
  icon: ReactNode
  label: string
  onClick?: () => void
}

export default function MainButton({ icon, label, onClick }: MainButtonProps) {
  return (
      <StyledCard onClick={onClick}>
        <IconLabelWrapper>
          {icon}
          <Label>{label}</Label>
        </IconLabelWrapper>
      </StyledCard>
  )
}

const StyledCard = styled.div`
  padding: 2rem;
  background-color: #dbeafe; 
  cursor: pointer;
  border-radius: 0.5rem;
  border: none;
  transition: all 0.2s ease-in-out;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);

  &:hover {
    background-color: #bfdbfe; 
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
  }
`

const IconLabelWrapper = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 1rem;
`

const Label = styled.span`
  font-size: 1.25rem;
  font-weight: 600;
  color: #1e3a8a;
`