'use client'

import { useState, useRef, useEffect } from "react"
import { User, LogOut, ChevronDown } from "lucide-react"
import styled from 'styled-components'
import { RoleBadgeType } from "@/types/RoleBadgeType"
import { KeyStorage } from "@/constants/KeyStorage"

export function UserDropdown({user, role, onClick}: {
    user: any
    role: string
    onClick: () => void
}) {
  const [isOpen, setIsOpen] = useState(false)
  const dropdownRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [])

  const getRoleText = (role: string) => {
    return role === KeyStorage.PROVIDER ? "관리자" : "참여자"
  }

  const getRoleBadgeColor = (role: string) => {
    return role === KeyStorage.PROVIDER ? { bg: '#f3e8ff', text: '#6b21a8' } : { bg: '#f3f4f6', text: '#1f2937' }
  }

  const roleStyle = getRoleBadgeColor(role) 
  const roleText = getRoleText(role)

 return (
    <Wrapper ref={dropdownRef}>
      <AvatarButton onClick={() => setIsOpen(!isOpen)}>
        <Avatar>
          <User className="icon" />
        </Avatar>
        <UserName>{user.user_metadata.username}</UserName>
        <ChevronDown className={`chevron ${isOpen ? 'open' : ''}`} />
      </AvatarButton>

      {isOpen && (
        <DropdownMenu>
          <UserInfo>
            <UserRow>
              <AvatarLarge>
                <User className="icon-lg" />
              </AvatarLarge>
              <UserDetails>
                <Name>{user.user_metadata.username}</Name>
                <Email>{user.email}</Email>
              </UserDetails>
            </UserRow>
            <RoleBadge bgColor={roleStyle.bg} textColor={roleStyle.text}>{roleText}</RoleBadge> 
          </UserInfo>

          <MenuItem>
            <button onClick={onClick}>
              <LogOut className="icon" />
              <span>로그아웃</span>
            </button>
          </MenuItem>
        </DropdownMenu>
      )}
    </Wrapper>
  )
}

const Wrapper = styled.div`
  position: relative;
`

const AvatarButton = styled.button`
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem;
  border-radius: 0.5rem;
  background: transparent;
  cursor: pointer;
  border: none;

  &:hover {
    background-color: #e5e7eb;
  }

  .chevron {
    width: 1rem;
    height: 1rem;
    color: #6b7280;
    transition: transform 0.2s;
  }

  .chevron.open {
    transform: rotate(180deg);
  }
`

const Avatar = styled.div`
  width: 2rem;
  height: 2rem;
  border-radius: 9999px;
  background-color: #2563eb;
  display: flex;
  align-items: center;
  justify-content: center;

  img {
    width: 100%;
    height: 100%;
    border-radius: 9999px;
    object-fit: cover;
  }

  .icon {
    width: 1rem;
    height: 1rem;
    color: white;
  }
`

const UserName = styled.span`
  display: none;
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;

  @media (min-width: 768px) {
    display: block;
  }
`

const DropdownMenu = styled.div`
  position: absolute;
  right: 0;
  margin-top: 0.5rem;
  width: 16rem;
  background-color: white;
  border: 1px solid #e5e7eb;
  border-radius: 0.5rem;
  box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
  z-index: 50;
  padding: 0.5rem 0;
`

const UserInfo = styled.div`
  padding: 0.75rem 1rem;
  border-bottom: 1px solid #f3f4f6;
`

const UserRow = styled.div`
  display: flex;
  align-items: center;
  gap: 0.75rem;
`

const AvatarLarge = styled.div`
  width: 2.5rem;
  height: 2.5rem;
  border-radius: 9999px;
  background-color: #2563eb;
  display: flex;
  align-items: center;
  justify-content: center;

  img {
    width: 100%;
    height: 100%;
    border-radius: 9999px;
    object-fit: cover;
  }

  .icon-lg {
    width: 1.25rem;
    height: 1.25rem;
    color: white;
  }
`

const UserDetails = styled.div`
  flex: 1;
  min-width: 0;
`

const Name = styled.p`
  font-size: 0.875rem;
  font-weight: 500;
  color: #111827;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`

const Email = styled.p`
  font-size: 0.75rem;
  color: #6b7280;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
`

const RoleBadge = styled.span<RoleBadgeType>`
  display: inline-flex;
  margin-top: 0.5rem;
  padding: 0.25rem 0.5rem;
  font-size: 0.75rem;
  font-weight: 500;
  border-radius: 9999px;
  background-color: ${({ bgColor }) => bgColor};
  color: ${({ textColor }) => textColor};
`

const MenuItem = styled.div`
  button {
    width: 100%;
    padding: 0.5rem 1rem;
    text-align: left;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: #374151;
    background: transparent;
    border: none;
    cursor: pointer;

    &:hover {
      background-color: #f9fafb;
    }

    .icon {
      width: 1rem;
      height: 1rem;
      color: #6b7280;
    }

    &.logout {
      color: #dc2626;

      &:hover {
        background-color: #fef2f2;
      }
    }
  }
`
