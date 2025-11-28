"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/utils/supabase/client";
import "../common/FedForm.css";
import { Path } from "@/constants/Path";

export default function AuthForm() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();
  const supabase = createClient();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    const { error } = await supabase.auth.signInWithPassword({
      email,
      password,
    });
    if (error) {
      alert("로그인에 실패했습니다. 다시 시도해 주세요.");
    } else {
      router.push(Path.DASHBOARD);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-wrapper">
        {" "}
        {/* ✅ 감싸는 래퍼 추가 */}
        <form onSubmit={handleLogin} className="auth-form">
          <h2 className="auth-title">로그인</h2>
          <input
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            autoComplete="username"
            className="auth-input"
          />
          <input
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            type="password"
            placeholder="Password"
            autoComplete="current-password"
            className="auth-input"
          />
          <button type="submit" className="auth-button">
            로그인
          </button>
        </form>
        <div className="signup-link" onClick={() => router.push(Path.SIGN_UP)}>
          회원가입하기
        </div>
      </div>
    </div>
  );
}
