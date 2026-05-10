"use server";

// =============================================================================
// SSR-only Server Action. Forwards Q&A questions to the FastAPI backend, which
// in turn calls OpenRouter server-side. The browser only sees this action's
// {success, answer} response — never the backend URL or the OpenRouter key.
// =============================================================================

import type { ChatMessage } from "@/lib/types";

const BACKEND_API_URL = process.env.BACKEND_API_URL ?? "http://localhost:8000";

type QnaResponse =
  | { success: true; answer: string }
  | { success: false; error: string };

export async function askAboutDiseaseAction(
  diseaseName: string,
  question: string,
  history: ChatMessage[] = []
): Promise<QnaResponse> {
  if (!diseaseName?.trim()) return { success: false, error: "Missing disease context." };
  if (!question?.trim()) return { success: false, error: "Question cannot be empty." };

  let res: Response;
  try {
    res = await fetch(`${BACKEND_API_URL}/qna`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        disease_name: diseaseName,
        question,
        history,
      }),
      cache: "no-store",
    });
  } catch (err) {
    console.error("Backend /qna unreachable:", err);
    return {
      success: false,
      error: `Backend not reachable at ${BACKEND_API_URL}. Start the FastAPI server.`,
    };
  }

  if (!res.ok) {
    let detail = `Backend returned HTTP ${res.status}.`;
    try {
      const json = (await res.json()) as { detail?: string };
      if (json.detail) detail = json.detail;
    } catch {
      /* ignore */
    }
    return { success: false, error: detail };
  }

  try {
    const json = (await res.json()) as { answer?: string };
    return { success: true, answer: (json.answer ?? "").trim() || "(empty response)" };
  } catch (err) {
    console.error("Failed to parse /qna response:", err);
    return { success: false, error: "Malformed response from backend." };
  }
}
