import { NextResponse } from "next/server";
import { cookies } from 'next/headers'
import { createClient } from '@/utils/supabase/server';
import { authorize } from "@/lib/supabase/authHelpers";

/** 소유 인스턴스 생성 */
export async function POST(request: Request) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase);
    if (!granted) return errRes;

    // DTO 언패킹
    const body = await request.json();
    const { name, ip_address, port } = body;
    const owner_id = user.id;

    // 인스턴스 생성
    const { data, error } = await supabase
        .from("instances")
        .insert([{ owner_id, name, ip_address, port }])
        .select().single();
    if (error) {
        console.log(error.message);
        return NextResponse.json({ error: "인스턴스 생성 중 문제가 발생했습니다." }, { status: 400 });
    }
    return NextResponse.json(data, { status: 201 });
}

/** 소유 인스턴스 조회 (목록 조회입니다!) */
export async function GET(request: Request) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase);
    if (!granted) return errRes;

    // 내 사용자 id
    const owner_id = user.id;

    // 내가 소유한 모든 인스턴스 조회
    const { data, error } = await supabase
        .from("instances")
        .select()
        .eq('owner_id', owner_id)
        .order('created_at', { ascending: true });
    if (error) {
        console.log(error.message);
        return NextResponse.json({ error: "인스턴스 조회 중 문제가 발생했습니다." }, { status: 400 });
    }
    return NextResponse.json(data);
}