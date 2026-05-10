"use server";

import { OpenAI } from "openai";

const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY ?? "";
const OPENROUTER_BASE_URL = process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1";
const OPENROUTER_TEXT_MODEL = process.env.OPENROUTER_TEXT_MODEL ?? "nvidia/nemotron-3-super-120b-a12b:free";

const client = new OpenAI({
  apiKey: OPENROUTER_API_KEY,
  baseURL: OPENROUTER_BASE_URL,
  defaultHeaders: {
    "HTTP-Referer": "http://localhost:3000",
    "X-Title": "LeafDoc-Frontend",
  },
});

export interface RecommendationResult {
  crops: {
    name: string;
    type: "Short-term" | "Long-term";
    suitabilityScore: number; // 0-100
    cultivationTime: string;
    description: string;
    benefits: string[];
  }[];
  cultivationGuide: string;
  bestPractices: string[];
}

export async function getRecommendationsAction(
  location: string,
  weather: any
): Promise<{ success: boolean; data?: RecommendationResult; error?: string }> {
  if (!OPENROUTER_API_KEY || OPENROUTER_API_KEY === "your_openrouter_api_key_here") {
    return {
      success: false,
      error: "OPENROUTER_API_KEY is not configured in the frontend .env.local.",
    };
  }

  const prompt = `
    Based on the location "${location}" and the following weather conditions: ${JSON.stringify(
    weather
  )},
    recommend suitable crops/plants to grow.

    Return ONLY a JSON object with this valid structure:
    {
      "crops": [
        {
          "name": "Crop Name",
          "type": "Short-term" or "Long-term",
          "suitabilityScore": number (0-100),
          "cultivationTime": "e.g., 3 months",
          "description": "Why it is suitable here",
          "benefits": ["benefit 1", "benefit 2"]
        }
      ],
      "cultivationGuide": "General detailed guide for this season/location",
      "bestPractices": ["practice 1", "practice 2"]
    }
  `;

  try {
    const response = await client.chat.completions.create({
      model: OPENROUTER_TEXT_MODEL,
      messages: [{ role: "user", content: prompt }],
      max_tokens: 2000,
      response_format: { type: "json_object" },
    });

    const text = response.choices[0]?.message?.content ?? "";
    const cleanText = text.replace(/```json\n|\n```/g, "").trim();
    const data = JSON.parse(cleanText) as RecommendationResult;

    return { success: true, data };
  } catch (error) {
    console.error("OpenRouter Recommendation Failed:", error);
    return { success: false, error: "Failed to generate recommendations." };
  }
}