import { NextRequest, NextResponse } from "next/server";
import { cookies } from 'next/headers'
import { createClient } from '@/utils/supabase/server';
import { authorize } from "@/lib/supabase/authHelpers";

/** 내 연합의 참여자 갱신 */
export async function PUT(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase, 'PROVIDER');
    if (!granted) return errRes;

    // DTO & URL 언패킹
    const body = await request.json();
    const { customer_id, instance_id = null } = body;
    const { id: participation_id } = await params;

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
        // URL 값을 토대로 참여 정보 갱신 시도
        const federation_id = fed.id;
        const { data, error } = await supabase
            .from("participations")
            .update({ customer_id, instance_id })
            .eq('federation_id', federation_id)
            .select().single();
        if (error) {
            console.log(error);
            return NextResponse.json({ error: "참여 정보 갱신 과정에서 문제가 발생했습니다." }, { status: 400 });
        }
        return NextResponse.json(data);
    }
    // 연합 없을 때
    else {
        return NextResponse.json({ error: "연합 정보가 없습니다." }, { status: 404 })
    }
}

/** 내 연합의 참여자 삭제 */
export async function DELETE(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase, 'PROVIDER');
    if (!granted) return errRes;

    // URL 파라미터 획득
    const { id: participation_id } = await params;

    // 삭제 시엔 연합 정보 굳이 조회하지 않고 바로 삭제함
    const { data, error } = await supabase.from("participations").delete().eq('id', participation_id);
    if (error) return NextResponse.json({ error: "참여 정보 삭제에 실패했습니다." }, { status: 400 });
    return new NextResponse(null, { status: 204 });
}