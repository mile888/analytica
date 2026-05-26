import { NextRequest, NextResponse } from "next/server";

const SESSION_COOKIE_NAME = "analytica_session_id";

function hasSessionCookie(request: NextRequest) {
  const value = request.cookies.get(SESSION_COOKIE_NAME)?.value;
  return Boolean(value && /^[0-9a-fA-F-]{36}$/.test(value));
}

export function middleware(request: NextRequest) {
  if (hasSessionCookie(request)) {
    return NextResponse.next();
  }

  const sessionId = crypto.randomUUID();
  const requestHeaders = new Headers(request.headers);
  const existingCookie = requestHeaders.get("cookie");
  const sessionCookie = `${SESSION_COOKIE_NAME}=${sessionId}`;
  requestHeaders.set("cookie", existingCookie ? `${existingCookie}; ${sessionCookie}` : sessionCookie);

  const response = NextResponse.next({
    request: {
      headers: requestHeaders,
    },
  });
  response.cookies.set(SESSION_COOKIE_NAME, sessionId, {
    httpOnly: true,
    sameSite: "lax",
    secure: false,
    path: "/",
  });
  return response;
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
