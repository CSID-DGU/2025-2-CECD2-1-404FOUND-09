import { NextResponse } from "next/server";
import { cookies } from 'next/headers'
import { createClient } from '@/utils/supabase/server';
import { authorize } from "@/lib/supabase/authHelpers";

/** 내 참여 갱신 API */
export async function PATCH(request: Request) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase, 'CUSTOMER');
    if (!granted) return errRes;

    // DTO 언패킹
    const body = await request.json();
    const { instance_id } = body;

    // instance_id 유효성 체크
    const { data: instance, error: instErr } = await supabase
        .from("instances").select('id').eq('id', instance_id).single();
    if (instErr || !instance) {
        return NextResponse.json({ error: "유효하지 않은 인스턴스입니다." }, { status: 400 });
    }

    // 내 참여 준비
    const customer_id = user.id;
    const { data: par, error: parErr } = await supabase.from("participations").select().eq('customer_id', customer_id).single();
    if (parErr) return NextResponse.json({ error: "내 참여 조회에 실패했습니다." }, { status: 400 });

    // 참여 있을 때
    if (par) {
        const { data, error } = await supabase.from("participations")
            .update({ instance_id })
            .eq('customer_id', customer_id)
            .select().single();
        if (error) {
            console.log(error);
            return NextResponse.json({ error: "참여 정보 갱신 과정에서 문제가 발생했습니다." }, { status: 400 });
        }
        return NextResponse.json(data);
    }
    // 참여 없을 때
    else {
        return NextResponse.json({ error: "참여 정보가 없습니다." }, { status: 404 });
    }
}

/** 내 참여 조회 API */
export async function GET(request: Request) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase, 'CUSTOMER');
    if (!granted) return errRes;

    // 내 참여 준비
    const customer_id = user.id;
    const { data: par, error: parErr } = await supabase.from("participations").select().eq('customer_id', customer_id).single();
    if (parErr) return NextResponse.json({ error: "내 참여 조회에 실패했습니다." }, { status: 400 });
    return NextResponse.json(par);
}