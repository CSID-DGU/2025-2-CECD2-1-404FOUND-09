import { NextResponse } from "next/server";
import { cookies } from 'next/headers'
import { createClient } from '@/utils/supabase/server';
import { authorize } from "@/lib/supabase/authHelpers";

/** 내 연합의 참여자 생성 */
export async function POST(request: Request) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase, 'PROVIDER');
    if (!granted) return errRes;

    // DTO 언패킹
    const body = await request.json();
    const { customer_id, instance_id = null } = body;

    // instance_id 가 지정된 경우, 유효성 체크
    if (instance_id !== null) {
        const { data: instance, error: instErr } = await supabase
            .from("instances").select('id').eq('id', instance_id).single();
        if (instErr || !instance) {
            return NextResponse.json({ error: "유효하지 않은 인스턴스입니다." }, { status: 400 });
        }
    }

    // 내 연합 준비
    const provider_id = user.id;
    const { data: fed, error: fedErr } = await supabase.from("federations").select().eq('provider_id', provider_id).single();
    if (fedErr) return NextResponse.json({ error: "연합 정보 조회에 실패했습니다." }, { status: 400 });

    // 연합 있을 때
    if (fed) {
        // 요청자가 이미 참여중이면 에러
        const { data: reserv } = await supabase.from("participations").select().eq('customer_id', customer_id).single();
        if (reserv) return NextResponse.json({ error: "이미 참여중인 사용자입니다." }, { status: 401 });

        // 아직 참여 안했으면 생성
        const federation_id = fed.id;
        const { data, error } = await supabase
            .from("participations")
            .insert([{ customer_id, federation_id, instance_id }])
            .select().single();
        if (error) {
            console.log(error);
            return NextResponse.json({ error: "참여 정보 생성 과정에서 문제가 발생했습니다." }, { status: 400 });
        }
        return NextResponse.json(data);
    }
    // 연합 없을 때
    else {
        return NextResponse.json({ error: "연합 정보가 없습니다." }, { status: 404 })
    }
}

/** 내 연합의 참여자 조회 */
export async function GET(request: Request) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase, 'PROVIDER');
    if (!granted) return errRes;

    // 내 연합 준비
    const provider_id = user.id;
    const { data: fed, error: fedErr } = await supabase.from("federations").select().eq('provider_id', provider_id).single();
    if (fedErr) return NextResponse.json({ error: "연합 정보 조회에 실패했습니다." }, { status: 400 });

    // 연합 있을 때
    if (fed) {
        const { data, error } = await supabase.rpc('get_customer_with_email', { federation_uuid: fed.id });
        if (error) {
            console.log(error.message);
            return NextResponse.json({ error: "연합의 참여 정보 조회에 실패했습니다." }, { status: 400 });
        }
        return NextResponse.json(data);
    }
    // 연합 없을 때
    else {
        return NextResponse.json({ error: "연합 정보가 없습니다." }, { status: 404 })
    }
}