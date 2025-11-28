import { NextResponse } from "next/server";
import { cookies } from 'next/headers'
import { createClient } from '@/utils/supabase/server';
import { authorize } from "@/lib/supabase/authHelpers";

/** 내 연합 생성(갱신) API */
export async function PUT(request: Request) {
	// supabase 연결 및 인가
	const cookieStore = cookies();
	const supabase = await createClient(cookieStore);
	const { granted, response: errRes, user } = await authorize(request, supabase, 'PROVIDER');
	if (!granted) return errRes;

	// 연합 없으면 만들고 있으면 갱신
	const body = await request.json();
	const { instance_id = null, name } = body;
	const provider_id = user.id;

	// instance_id 가 지정된 경우, 유효성 체크
	if (instance_id !== null) {
		const { data: instance, error: instErr } = await supabase
			.from("instances").select('id').eq('id', instance_id).single();
		if (instErr || !instance) {
			return NextResponse.json({ error: "유효하지 않은 인스턴스입니다." }, { status: 400 });
		}
	}

	// 이미 provider_id == user.id인 federation이 있는지 확인
	const { data: fed, error: fedErr } = await supabase
		.from("federations").select().eq('provider_id', provider_id).single();
	if (fed) {
		// 있으면 갱신
		const { data, error } = await supabase
			.from("federations")
			.update({ instance_id, name })
			.eq('provider_id', provider_id)
			.select().single();
		if (error) {
			console.log(error.message);
			return NextResponse.json({ error: "연합 정보 갱신 과정에서 문제가 발생했습니다." }, { status: 400 });
		}
		return NextResponse.json(data);
	} else {
		// 없으면 provider_id 지정해서 생성
		const { data, error } = await supabase
			.from("federations")
			.insert([{ provider_id, instance_id, name }])
			.select().single();
		if (error) {
			console.log(error.message);
			return NextResponse.json({ error: "연합 정보 생성 과정에서 문제가 발생했습니다." }, { status: 400 });
		}
		return NextResponse.json(data);
	}
}

/** 내 연합 조회 API */
export async function GET(request: Request) {
	// supabase 연결 및 인가
	const cookieStore = cookies();
	const supabase = await createClient(cookieStore);
	const { granted, response: errRes, user } = await authorize(request, supabase, 'PROVIDER');
	if (!granted) return errRes;

	// 연합 조회
	const { data, error } = await supabase.from("federations").select().eq('provider_id', user.id).single();
	if (error) return new NextResponse(null, { status: 204 });
	return NextResponse.json(data);
}