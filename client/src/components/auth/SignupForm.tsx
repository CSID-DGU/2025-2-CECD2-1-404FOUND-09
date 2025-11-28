"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/utils/supabase/client";
import "../common/FedForm.css";
import { Path } from "@/constants/Path";

export default function SignupForm() {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const router = useRouter();
  const supabase = createClient();

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();

    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          username: username, // 추가 정보 저장
        },
      },
    });

    if (error) {
      alert("회원가입에 실패했습니다.");
    } else {
      alert("회원가입이 완료되었습니다. 이메일을 확인해 주세요.");
      router.push(Path.LOGIN);
    }
  };

  return (
    <div className="auth-container">
      <form onSubmit={handleSignup} className="auth-form">
        <h2 className="auth-title">회원가입</h2>
        <input
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          placeholder="Username"
          autoComplete="username"
          className="auth-input"
        />
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          autoComplete="email"
          className="auth-input"
        />
        <input
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          type="password"
          placeholder="Password"
          autoComplete="new-password"
          className="auth-input"
        />
        <button type="submit" className="auth-button">
          회원가입
        </button>
      </form>
    </div>
  );
}
