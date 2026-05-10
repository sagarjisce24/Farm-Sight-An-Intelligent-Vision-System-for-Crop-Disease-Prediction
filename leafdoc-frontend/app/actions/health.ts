"use server";

// SSR-only server action. Keeps the BACKEND_API_URL off the client.

import type { BackendHealth } from "@/lib/types";

const BACKEND_API_URL = process.env.BACKEND_API_URL ?? "http://localhost:8000";

export async function checkBackendHealthAction(): Promise<BackendHealth> {
  try {
    const res = await fetch(`${BACKEND_API_URL}/health`, {
      cache: "no-store",
      // Short timeout so the UI doesn't hang
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) {
      return {
        reachable: false,
        modelLoaded: false,
        leafModelLoaded: false,
        classesCount: 0,
        diseaseInfoLoaded: false,
        error: `HTTP ${res.status}`,
      };
    }
    const json = (await res.json()) as {
      model_loaded: boolean;
      leaf_model_loaded: boolean;
      classes_count: number;
      disease_info_loaded: boolean;
      thresholds?: { leaf: number; confidence: number; entropy: number };
    };
    return {
      reachable: true,
      modelLoaded: json.model_loaded,
      leafModelLoaded: json.leaf_model_loaded,
      classesCount: json.classes_count,
      diseaseInfoLoaded: json.disease_info_loaded,
      thresholds: json.thresholds,
    };
  } catch (err) {
    return {
      reachable: false,
      modelLoaded: false,
      leafModelLoaded: false,
      classesCount: 0,
      diseaseInfoLoaded: false,
      error: err instanceof Error ? err.message : "Unknown error",
    };
  }
}
