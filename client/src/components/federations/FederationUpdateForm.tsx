"use client";

import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import "../common/FedForm.css";
import { Path } from "@/constants/Path";
import { InstanceType } from "@/types/InstanceType";

export default function FederationUpdateForm() {
  const [name, setName] = useState("");
  const [instanceId, setInstanceId] = useState("");
  const [instances, setInstances] = useState<InstanceType[]>([]); // dropdown 데이터
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const queryName = searchParams.get("name") || "";
    const queryInstanceId = searchParams.get("instance_id");
    setName(queryName);
    if (queryInstanceId && queryInstanceId !== "(없음)") {
      setInstanceId(queryInstanceId);
    }
  }, [searchParams]);

  useEffect(() => {
    const fetchInstances = async () => {
      try {
        const res = await fetch("/api/instances");
        const data = await res.json();
        if (res.ok) {
          setInstances(data as InstanceType[]);
        } else {
          console.error("인스턴스 목록 조회 실패");
        }
      } catch (err) {
        console.error("네트워크 오류:", err);
      }
    };
    fetchInstances();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      alert("이름(name)은 필수입니다.");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch("/api/federation", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          instance_id: instanceId || null,
        }),
      });
      const result = await res.json();

      if (res.status === 200 || res.status === 201) {
        router.push(Path.DASHBOARD);
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
        <h2 className="auth-title">연합 정보 업데이트</h2>

        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="이름 (필수)"
          autoComplete="name"
          className="auth-input"
        />

        <select
          value={instanceId}
          onChange={(e) => setInstanceId(e.target.value)}
          className="form-select"
        >
          <option value="">인스턴스를 선택하세요 (선택)</option>
          {instances.map((instance) => (
            <option key={instance.id} value={instance.id}>
              {`${instance.name} (Ip Address: ${instance.ip_address ?? "없음"}, Port: ${instance.port})`}
            </option>
          ))}
        </select>

        <button type="submit" disabled={loading} className="auth-button">
          {loading ? "처리 중..." : "저장"}
        </button>
      </form>
    </div>
  );
}
