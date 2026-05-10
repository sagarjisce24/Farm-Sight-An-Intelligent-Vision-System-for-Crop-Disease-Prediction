"use server";

// =============================================================================
// SSR-only. This file runs exclusively on the Next.js server.
// - OPENROUTER_API_KEY and BACKEND_API_URL are read from server-side env vars.
// - The browser never imports this file (Server Action boundary enforces it).
// - DO NOT import this file from any client component.
// =============================================================================

import { OpenAI } from "openai";
import type { AnalysisProvider, AnalysisResult } from "@/lib/types";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY ?? "";
const OPENROUTER_BASE_URL = process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1";
const OPENROUTER_VISION_MODEL = process.env.OPENROUTER_VISION_MODEL ?? "qwen/qwen2.5-vl-72b-instruct:free";
const BACKEND_API_URL = process.env.BACKEND_API_URL ?? "http://localhost:8000";

type ActionResponse =
  | { success: true; data: AnalysisResult }
  | { success: false; error: string };

export async function analyzeImageAction(formData: FormData): Promise<ActionResponse> {
  const file = formData.get("file");
  const provider = (formData.get("provider") as AnalysisProvider | null) ?? "openrouter";

  if (!file || !(file instanceof File)) {
    return { success: false, error: "No file provided." };
  }

  if (provider === "custom") {
    return analyzeWithCustomBackend(file);
  }
  return analyzeWithOpenRouter(file);
}

// -----------------------------------------------------------------------------
// Custom FastAPI backend
// -----------------------------------------------------------------------------

async function analyzeWithCustomBackend(file: File): Promise<ActionResponse> {
  const upstream = new FormData();
  upstream.append("file", file, file.name || "leaf.jpg");

  let res: Response;
  try {
    res = await fetch(`${BACKEND_API_URL}/predict`, {
      method: "POST",
      body: upstream,
      cache: "no-store",
    });
  } catch (err) {
    console.error("Backend /predict unreachable:", err);
    return {
      success: false,
      error:
        "Could not reach the custom-model backend. Is the FastAPI server running on " +
        `${BACKEND_API_URL}? Try the OpenRouter provider instead.`,
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

  const json = (await res.json()) as Record<string, unknown>;
  // The backend already returns the AnalysisResult shape minus `provider`.
  // Cast carefully and stamp the provider so downstream UI can show a badge.
  const data = { ...(json as object), provider: "custom" as const } as AnalysisResult;
  return { success: true, data };
}

// -----------------------------------------------------------------------------
// OpenRouter vision (server-side only — SDK is never bundled into the client)
// -----------------------------------------------------------------------------

async function analyzeWithOpenRouter(file: File): Promise<ActionResponse> {
  if (!OPENROUTER_API_KEY || OPENROUTER_API_KEY === "your_openrouter_api_key_here") {
    return {
      success: false,
      error:
        "OPENROUTER_API_KEY is not configured in the frontend .env.local. " +
        "Either set it or use the Custom Model provider.",
    };
  }

  try {
    const arrayBuffer = await file.arrayBuffer();
    const base64 = Buffer.from(arrayBuffer).toString("base64");

    const client = new OpenAI({
      apiKey: OPENROUTER_API_KEY,
      baseURL: OPENROUTER_BASE_URL,
      defaultHeaders: {
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "LeafDoc-Frontend",
      },
    });

    const prompt = `
      Analyze this plant image for diseases.
      Return ONLY a JSON object with this structure:
      {
        "isHealthy": boolean,
        "diseaseName": "string or 'Healthy'",
        "confidence": number (0-100),
        "description": "brief description of condition",
        "symptoms": ["list", "of", "observable", "symptoms"],
        "treatment": ["step by step", "treatment", "instructions"],
        "prevention": ["prevention", "tips"],
        "severity": number (0-100, 0 if healthy),
        "progression": [
          {"stage": "Early", "timeline": "Days 1-3"},
          {"stage": "Mid", "timeline": "Days 4-7"},
          {"stage": "Late", "timeline": "Week 2+"}
        ],
        "environmentalFactors": {
          "temperature": "ideal range",
          "humidity": "ideal range",
          "sunlight": "requirements",
          "watering": "requirements"
        }
      }
    `;

    const response = await client.chat.completions.create({
      model: OPENROUTER_VISION_MODEL,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: prompt },
            {
              type: "image_url",
              image_url: {
                url: `data:${file.type || "image/jpeg"};base64,${base64}`,
                detail: "high",
              },
            },
          ],
        },
      ],
      max_tokens: 2000,
      response_format: { type: "json_object" },
    });

    const text = response.choices[0]?.message?.content ?? "";
    const cleaned = text.replace(/```json\n|\n```/g, "").trim();
    const parsed = JSON.parse(cleaned) as Omit<AnalysisResult, "status" | "provider">;

    const data: AnalysisResult = {
      ...parsed,
      status: "ok",
      provider: "openrouter",
    };
    return { success: true, data };
  } catch (err) {
    console.error("OpenRouter vision analysis failed:", err);
    return { success: false, error: "Failed to analyze image with OpenRouter. Please try again." };
  }
}