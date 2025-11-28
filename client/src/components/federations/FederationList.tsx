"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import "../common/FedForm.css";
import "./FederationList.css";
import LoadingView from "../common/LoadingView";
import ErrorView from "../common/ErrorView";
import { toast } from "sonner";
import { Path } from "@/constants/Path";
import { SquarePen, Trash } from "lucide-react";

interface Instance {
  id: number;
  name: string;
  ip_address: number;
  port: number;
}

export default function FederationList() {
  const [instances, setInstances] = useState<Instance[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  useEffect(() => {
    // GET 요청 수행
    const fetchInstances = async () => {
      setLoading(true);
      try {
        const res = await fetch("/api/instances");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        setInstances(data);
      } catch (e: any) {
        setError(e.message || "조회 실패");
      }
      setLoading(false);
    };

    fetchInstances();
  }, []);

  const handleDelete = async (id: number) => {
    const ok = window.confirm("정말 삭제하시겠습니까?");
    if (!ok) return;
    setLoading(true);
    try {
      const res = await fetch(`/api/instances/${id}`, {
        method: "DELETE",
      });

      if (!res.ok) throw new Error("삭제 실패");

      // 삭제 성공 시 상태에서 해당 인스턴스 제거
      setInstances((prev) => prev.filter((instance) => instance.id !== id));
    } catch (error) {
      console.error("삭제 요청 실패:", error);
      alert("등록되어 있는 인스턴스는 삭제할 수 없습니다.");
    }
    setLoading(false);
  };

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
    <div className="instance-container">
      <table className="instance-table">
        <thead>
          <tr>
            <th>Name</th>
            <th>IpAddress</th>
            <th>Port</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {instances.map((instance) => (
            <tr key={instance.id}>
              <td className="instance-name">{instance.name}</td>
              <td className="instance-ip">{instance.ip_address}</td>
              <td className="instance-port">{instance.port}</td>
              <td>
                <div className="instance-actions">
                  <SquarePen
                    color="#010101"
                    onClick={() =>
                      router.push(`/instance/create?id=${instance.id}`)
                    }
                  />
                  <Trash
                    color="#010101"
                    onClick={() => handleDelete(instance.id)}
                  />
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <button
        className="auth-button"
        onClick={() => router.push(Path.INSTANCE_CREATE)}
      >
        추가하기
      </button>
    </div>
  );
}
