import { NextResponse } from "next/server";
import { cookies } from 'next/headers'
import { createClient } from '@/utils/supabase/server';
import { authorize } from "@/lib/supabase/authHelpers";

/** 내 역할 조회 API */
export async function GET(request: Request) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase);
    if (!granted) return errRes;

    // 유저 조회
    const { data, error } = await supabase.from("profiles").select('service_role').eq('id', user.id).single();
    if (error) {
        console.log(error.message);
        return NextResponse.json({ error: "내 역할 조회에 실패했습니다." }, { status: 400 });
    }
    return NextResponse.json(data);
}