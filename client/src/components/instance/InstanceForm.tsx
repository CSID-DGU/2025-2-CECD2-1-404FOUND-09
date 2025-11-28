"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import "../common/FedForm.css";
import { Path } from "@/constants/Path";

export default function InstanceForm() {
  const [name, setName] = useState("");
  const [ip_address, setIpAddress] = useState("");
  const [port, setPort] = useState("");
  const [loading, setLoading] = useState(false);

  const router = useRouter();
  const searchParams = useSearchParams();
  const id = searchParams.get("id"); // ?id=... 이 있으면 수정 모드
  const isEdit = !!id;

  useEffect(() => {
    if (!isEdit) return;
    const fetchInstance = async () => {
      try {
        const res = await fetch(`/api/instances/${id}`);
        if (!res.ok) throw new Error("인스턴스 정보를 불러오지 못했습니다.");
        const data = await res.json();
        setName(data.name ?? "");
        setIpAddress(String(data.ip_address ?? ""));
        setPort(String(data.port ?? ""));
      } catch (e) {
        alert("데이터 로딩 실패");
        router.push(Path.INSTANCE);
      }
    };
    fetchInstance();
  }, [isEdit, id, router]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const res = await fetch(
        isEdit ? `/api/instances/${id}` : "/api/instances",
        {
          method: isEdit ? "PUT" : "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, ip_address, port }),
        }
      );
      const result = await res.json();

      if (res.status === 200 || res.status === 201) {
        alert(
          isEdit ? "수정이 완료되었습니다." : "인스턴스 생성이 완료되었습니다."
        );
        router.push(Path.INSTANCE);
      } else {
        alert("오류: " + (result?.error || "알 수 없는 오류"));
      }
    } catch (err: any) {
      alert("네트워크 오류: " + (err.message || err));
    }
    setLoading(false);
  };

  return (
    <div className="auth-container">
      <form onSubmit={handleSubmit} className="auth-form">
        <h2 className="auth-title">
          {isEdit ? "인스턴스 수정" : "인스턴스 생성"}
        </h2>
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Name"
          autoComplete="name"
          className="auth-input"
        />
        <input
          value={ip_address}
          onChange={(e) => setIpAddress(e.target.value)}
          placeholder="IpAddress"
          autoComplete="ipAddress"
          className="auth-input"
        />
        <input
          value={port}
          onChange={(e) => setPort(e.target.value)}
          type="port"
          placeholder="Port"
          autoComplete="new-port"
          className="auth-input"
        />
        <button type="submit" disabled={loading} className="auth-button">
          {loading ? "처리 중..." : isEdit ? "수정하기" : "생성하기"}
        </button>
      </form>
    </div>
  );
}
