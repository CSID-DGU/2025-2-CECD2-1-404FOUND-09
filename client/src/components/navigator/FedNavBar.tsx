"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { createClient } from "@/utils/supabase/client";
import { UserDropdown } from "./UserDropDown";
import { KeyStorage } from "@/constants/KeyStorage";
import { Path } from "@/constants/Path";
import { useRouter } from "next/navigation";

export default function FedNavBar() {
  const [user, setUser] = useState<any>(null);
  const [role, setRole] = useState("");
  const supabase = createClient();
  const router = useRouter();

  useEffect(() => {
    supabase.auth
      .getUser()
      .then(({ data, error }) => {
        if (error) throw error;
        setUser(data.user);
      })
      .catch(() => setUser(null));

    const { data: listener } = supabase.auth.onAuthStateChange(
      (event, session) => {
        if (event === KeyStorage.SIGNED_IN) {
          setUser(session?.user ?? null);
        } else if (event === KeyStorage.SIGNED_OUT) {
          setUser(null);
        }
      }
    );

    // 컴포넌트 언마운트 시 cleanup
    return () => {
      listener?.subscription.unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (!user) return;

    const fetchServiceRole = async () => {
      try {
        const res = await fetch("/api/me");
        if (!res.ok) throw new Error("로그인 필요");
        const data = await res.json();
        setRole(data.service_role);
      } catch (e: any) {
        console.log(e.message || "조회 실패");
      }
    };
    fetchServiceRole();
  }, [user]);

  const handleLogout = async () => {
    const { error } = await supabase.auth.signOut();
    if (error) {
      console.error("로그아웃 실패:", error);
    } else {
      router.push(Path.MAIN);
    }
  };

  return (
    <nav>
      <div className="container">
        <div className="logo-group">
          <Link href={Path.MAIN}>
            <h1>
              <span className="logo-f">Fed</span>
              <span className="logo-h">H</span>
              <span className="logo-b">B</span>
            </h1>
          </Link>
        </div>
        <div className="nav-menu">
          <Link href={Path.ABOUT}>FedHB 소개</Link>
          <span className="nav-divider">|</span>
          <Link href={Path.DASHBOARD}>대시보드</Link>
          <span className="nav-divider">|</span>
          <Link href={Path.INSTANCE}>인스턴스</Link>
          <span className="nav-divider">|</span>
          {user ? (
            <UserDropdown user={user} role={role} onClick={handleLogout} />
          ) : (
            <Link href={Path.LOGIN}>로그인</Link>
          )}
        </div>
      </div>
    </nav>
  );
}
