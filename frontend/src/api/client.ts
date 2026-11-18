import { localeTag } from "../i18n";

// Thin fetch wrapper. The backend always answers errors as {"error": "..."}; that text is
// written for people, so it is shown as is. Accept-Language carries the language picked in
// the UI, so scenario text and Setup checks come back in it (docs/architecture.md §3.7).
export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

export async function api<T>(path: string, init?: { method?: string; body?: unknown }): Promise<T> {
  const headers: Record<string, string> = { "Accept-Language": localeTag() };
  if (init?.body !== undefined) headers["Content-Type"] = "application/json";
  const res = await fetch(path, {
    method: init?.method ?? (init?.body !== undefined ? "POST" : "GET"),
    headers,
    body: init?.body !== undefined ? JSON.stringify(init.body) : undefined,
    credentials: "same-origin",
  });
  const text = await res.text();
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    throw new ApiError(`${path} returned ${res.status} with a body that is not JSON`, res.status);
  }
  if (!res.ok) {
    const message = (data as { error?: string; detail?: unknown })?.error
      ?? JSON.stringify((data as { detail?: unknown })?.detail ?? data);
    throw new ApiError(message, res.status);
  }
  return data as T;
}
