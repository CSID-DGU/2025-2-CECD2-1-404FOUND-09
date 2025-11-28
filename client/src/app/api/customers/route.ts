import { NextResponse } from "next/server";
import { cookies } from 'next/headers'
import { createClient } from '@/utils/supabase/server';
import { authorize } from "@/lib/supabase/authHelpers";

/** 전체 참여자 조회 API */
export async function GET(request: Request) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase, 'PROVIDER');
    if (!granted) return errRes;

    // 유저 조회
    const { data, error } = await supabase.rpc('get_all_customers');
    if (error) {
        console.log(error.message);
        return NextResponse.json({ error: "전체 사용자 조회에 실패했습니다." }, { status: 400 });
    }
    return NextResponse.json(data);
}