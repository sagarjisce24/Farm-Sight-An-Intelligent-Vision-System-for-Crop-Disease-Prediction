"use server";

// SSR-only server action — fetches the supported-class list from FastAPI.

import type { SupportedClasses } from "@/lib/types";

const BACKEND_API_URL = process.env.BACKEND_API_URL ?? "http://localhost:8000";

type Result =
  | { success: true; data: SupportedClasses }
  | { success: false; error: string };

export async function getSupportedClassesAction(): Promise<Result> {
  try {
    const res = await fetch(`${BACKEND_API_URL}/supported-classes`, {
      cache: "no-store",
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return { success: false, error: `HTTP ${res.status}` };

    const json = (await res.json()) as {
      total_classes: number;
      species_count: number;
      by_species: Record<string, string[]>;
      limitations: string[];
      fallback_advice: string;
    };

    return {
      success: true,
      data: {
        totalClasses: json.total_classes,
        speciesCount: json.species_count,
        bySpecies: json.by_species,
        limitations: json.limitations,
        fallbackAdvice: json.fallback_advice,
      },
    };
  } catch (err) {
    return {
      success: false,
      error: err instanceof Error ? err.message : "Failed to fetch supported classes.",
    };
  }
}
