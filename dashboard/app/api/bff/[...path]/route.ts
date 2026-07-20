// dashboard/app/api/bff/[...path]/route.ts
// E-CIP v3.0 — BFF Proxy
// Blueprint Section 12 — Fix #23
//
// Every dashboard API call goes through this route handler, which runs
// server-side in Next.js (never in the browser). The API key lives only
// in the server environment (ECIP_API_KEY) and is attached here — the
// browser never sees it, never sends it, and it never appears in any
// client-side network request the user's devtools could inspect.

import { NextRequest, NextResponse } from "next/server";

const API_BASE_URL = process.env.ECIP_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.ECIP_API_KEY ?? "";

async function proxy(request: NextRequest, path: string[]): Promise<NextResponse> {
  const targetUrl = `${API_BASE_URL}/${path.join("/")}${request.nextUrl.search}`;

  const headers = new Headers();
  headers.set("X-API-Key", API_KEY);
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  const init: RequestInit = {
    method: request.method,
    headers,
  };

  if (request.method !== "GET" && request.method !== "HEAD") {
    init.body = await request.text();
  }

  try {
    const response = await fetch(targetUrl, init);
    const body = await response.text();
    return new NextResponse(body, {
      status: response.status,
      headers: { "content-type": response.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return NextResponse.json(
      { error: "E-CIP API unreachable", target: targetUrl },
      { status: 502 }
    );
  }
}

export async function GET(request: NextRequest, { params }: { params: { path: string[] } }) {
  return proxy(request, params.path);
}

export async function POST(request: NextRequest, { params }: { params: { path: string[] } }) {
  return proxy(request, params.path);
}

export async function PUT(request: NextRequest, { params }: { params: { path: string[] } }) {
  return proxy(request, params.path);
}

export async function DELETE(request: NextRequest, { params }: { params: { path: string[] } }) {
  return proxy(request, params.path);
}
