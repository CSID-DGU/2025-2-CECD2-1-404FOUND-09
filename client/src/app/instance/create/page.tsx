'use client'

import { Suspense } from "react";
import InstanceForm from "@/components/instance/InstanceForm";

export default function InstanceCreate() {
  return (
    <main>
      <Suspense fallback={<div>로딩 중...</div>}>
        <InstanceForm />
      </Suspense>
    </main>
  )
}
