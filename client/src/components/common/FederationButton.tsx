'use client'

import styled from 'styled-components'
import React from 'react'

interface ButtonProps {
  label: string
  onClick: () => void
  icon?: React.ReactNode
  bgColor?: string
  textColor?: string
  fontSize?: string
}

const StyledButton = styled.button<{
  $bgColor: string
  $textColor: string
  $fontSize: string
}>`
  background-color: ${({ $bgColor }) => $bgColor};
  color: ${({ $textColor }) => $textColor};
  font-size: ${({ $fontSize }) => $fontSize};
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  border: none;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-weight: 500;
  transition: background-color 0.2s ease;

  &:hover {
    filter: brightness(0.9);
  }

  svg {
    width: 1rem;
    height: 1rem;
  }
`

export default function FederationButton({
  icon,
  label,
  onClick,
  bgColor = '#2563eb',    
  textColor = '#ffffff',
  fontSize = '0.875rem',
}: ButtonProps) {
  return (
    <StyledButton
      onClick={onClick}
      $bgColor={bgColor}
      $textColor={textColor}
      $fontSize={fontSize}
    >
      {icon}
      <span>{label}</span>
    </StyledButton>
  )
}
