import { NextRequest, NextResponse } from "next/server";
import { cookies } from 'next/headers'
import { createClient } from '@/utils/supabase/server';
import { authorize } from "@/lib/supabase/authHelpers";

/** 소유 인스턴스 갱신 전 정보 불러오기 */
export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const cookieStore = cookies();
  const supabase = await createClient(cookieStore);
  const { granted, response: errRes, user } = await authorize(request, supabase);
  if (!granted) return errRes;

  const { id } = await params;

  const { data, error } = await supabase
    .from("instances")
    .select()
    .eq("id", id)
    .eq("owner_id", user.id) // 소유자 검증
    .single();

  if (error) {
    console.error(error.message);
    return NextResponse.json(
      { error: "인스턴스 정보를 불러오지 못했습니다." },
      { status: 404 }
    );
  }

  return NextResponse.json(data, { status: 200 });
}

/** 소유 인스턴스 갱신 */
export async function PUT(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase);
    if (!granted) return errRes;

    // DTO & URL 언패킹
    const body = await request.json();
    const { name, ip_address, port } = body;
    const owner_id = user.id;
    const { id } = await params;

    // 인스턴스 갱신
    const { data, error } = await supabase
        .from("instances")
        .update([{ name, ip_address, port }])
        .eq('owner_id', owner_id)
        .eq('id', id)
        .select().single();
    if (error) {
        console.log(error.message);
        return NextResponse.json({ error: "인스턴스 갱신 중 문제가 발생했습니다." }, { status: 400 });
    }
    return NextResponse.json(data);
}

/** 소유 인스턴스 삭제 */
export async function DELETE(request: NextRequest, { params }: { params: Promise<{ id: string }> }) {
    // supabase 연결 및 인가
    const cookieStore = cookies();
    const supabase = await createClient(cookieStore);
    const { granted, response: errRes, user } = await authorize(request, supabase);
    if (!granted) return errRes;

    // URL 파라미터 획득
    const { id } = await params;

    // 인스턴스 삭제
    const { data, error } = await supabase.from("instances").delete().eq('id', id);
    if (error) return NextResponse.json({ error: "인스턴스 삭제에 실패했습니다." }, { status: 400 });
    return new NextResponse(null, { status: 204 });
}