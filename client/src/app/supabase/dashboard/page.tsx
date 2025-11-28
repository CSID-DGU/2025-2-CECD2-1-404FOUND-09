"use client";

import styled from "styled-components";
import { useEffect, useState } from "react";
import LoadingView from "@/components/common/LoadingView";
import ErrorView from "@/components/common/ErrorView";
import "@/components/common/DashboardCard.css";
import { KeyStorage } from "@/constants/KeyStorage";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Path } from "@/constants/Path";
import ProviderDashboard from "@/components/Dashboard/ProviderDashboard";
import CustomerDashboard from "@/components/Dashboard/CustomerDashboard";

export default function Home() {
  const [role, setRole] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    const fetchServiceRole = async () => {
      setLoading(true);
      try {
        // 타임아웃 설정 (5초)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        const res = await fetch("/api/me", {
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        setRole(data.service_role);
      } catch (e: any) {
        if (e.name === 'AbortError') {
          setError("서버 응답 시간이 초과되었습니다.");
        } else {
          setError(e.message || "조회 실패");
        }
        console.error('사용자 역할 조회 실패:', e);
      }
      setLoading(false);
    };
    
    fetchServiceRole();
  }, []);

  if (loading) {
    return <LoadingView />;
  }

  if (error) {
    if (error.includes("로그인")) {
      toast.error(error);
      router.push(Path.LOGIN);
      return null;
    }

    return <ErrorView message={error} />;
  }

  return (
    <Container>
      <div style={{ minHeight: '100vh' }}>
        {role === KeyStorage.PROVIDER ? (
          <ProviderDashboard />
        ) : (
          <CustomerDashboard />
        )}
      </div>
    </Container>
  );
}

const Container = styled.div`
  background-color: white;
  height: 100%;
  padding: 2rem;
`;
