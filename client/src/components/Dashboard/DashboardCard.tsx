'use client'

import React, { useEffect, useState } from 'react'
import '../common/DashboardCard.css';

interface DashboardData {
  total_client: number
  accuracy: number
  cost: number
}

const dummyDashboardData: DashboardData = {
  total_client: 50,
  accuracy: 75,
  cost: 103.7
}

export default function DashboardCard() {
  const [data, setData] = useState<DashboardData | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // /api/dashboard 엔드포인트가 존재하지 않으므로 더미 데이터 사용
        console.log('대시보드 API 엔드포인트가 없어서 더미 데이터를 사용합니다.');
        throw new Error('Dashboard API not implemented');
      } catch (e) {
        console.error('API 실패, 더미로 대체:', e);
        setData(dummyDashboardData); // 에러 → 더미 사용
      }
    };

    fetchData();
  }, []);


  return (
    <div className="card-wrapper">
      <div className="card-grid">
        <div className="card">
          <div className="card-title">Total Clients</div>
          <div className="card-value">{data?.total_client ?? '-'}</div>
        </div>
        <div className="card">
          <div className="card-title">Current Round</div>
          <div className="card-value">10</div> {/* 예시로 하드코딩 */}
        </div>
        <div className="card">
          <div className="card-title">Final Accuracy</div>
          <div className="card-value">{data ? `${data.accuracy}%` : '-'}</div>
        </div>
        <div className="card">
          <div className="card-title">Communication Cost</div>
          <div className="card-value">{data ? `${data.cost.toFixed(1)}MB` : '-'}</div>
        </div>
      </div>
    </div>
  )
}
