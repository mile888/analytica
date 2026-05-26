/**
 * Streaming upload proxy for CSV dataset uploads.
 *
 * Next.js rewrites buffer the request body through the internal proxy,
 * which enforces a default ~10 MB body limit.  For dataset uploads that
 * can reach 50+ MB, we need a dedicated Route Handler that streams the
 * request body directly to the FastAPI backend without buffering.
 *
 * This route takes priority over the catch-all rewrite because Next.js
 * resolves explicit Route Handlers before rewrites.
 */

import { NextRequest, NextResponse } from "next/server";

/** Maximum upload size in bytes (100 MB). */
const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

/** Backend URL resolved at runtime. */
function backendUrl(): string {
  return process.env.INTERNAL_API_URL || "http://backend:8000";
}

export async function POST(request: NextRequest) {
  const contentLength = request.headers.get("content-length");
  if (contentLength && parseInt(contentLength, 10) > MAX_UPLOAD_BYTES) {
    return NextResponse.json(
      { detail: `Upload too large. Maximum size is ${MAX_UPLOAD_BYTES / 1024 / 1024} MB.` },
      { status: 413 },
    );
  }

  const targetUrl = `${backendUrl()}/data-sources/upload-csv`;

  try {
    const headers = new Headers();
    const contentType = request.headers.get("content-type");
    if (contentType) {
      headers.set("content-type", contentType);
    }
    if (contentLength) {
      headers.set("content-length", contentLength);
    }
    const cookie = request.headers.get("cookie");
    if (cookie) {
      headers.set("cookie", cookie);
    }

    const backendResponse = await fetch(targetUrl, {
      method: "POST",
      headers,
      body: request.body,
      cache: "no-store",
      next: { revalidate: 0 },
      // @ts-expect-error -- duplex is required for streaming request bodies in Node 18+
      duplex: "half",
    });

    const responseHeaders = new Headers();
    backendResponse.headers.forEach((value, key) => {
      // Skip hop-by-hop headers
      if (!["transfer-encoding", "connection"].includes(key.toLowerCase())) {
        responseHeaders.set(key, value);
      }
    });

    return new NextResponse(backendResponse.body, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: responseHeaders,
    });
  } catch (error) {
    console.error("[upload-csv proxy] Backend request failed:", error);
    return NextResponse.json(
      {
        detail: "Upload proxy failed: could not reach the backend service.",
        error: error instanceof Error ? error.message : String(error),
      },
      { status: 502 },
    );
  }
}
