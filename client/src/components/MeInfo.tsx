'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation'
import { createClient } from '@/utils/supabase/client';

export default function MeInfo() {
  const [user, setUser] = useState<any>(null);
  const router = useRouter()
  const supabase = createClient();

  useEffect(() => {
    supabase.auth.getUser()
      .then(({ data, error }) => {
        if (error) throw error;
        setUser(data.user);
      })
      .catch(err => {
        console.log(err);
        setUser(null);
      });
  }, []);

  const handleLogout = async () => {
    await supabase.auth.signOut()
    router.push('/supabase/login')
  }

  return (
    <div>
      <h2>내 정보</h2>
      <pre>{user ?
        <>
          {JSON.stringify(user, null, 2)}
          <button onClick={handleLogout}>로그아웃</button>
        </>
        : '로그인 필요'}</pre>
    </div>
  );
}
