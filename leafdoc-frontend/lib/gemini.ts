// Shared types used by both Gemini and Custom-Model code paths.
// IMPORTANT: this file contains TYPES ONLY. Do NOT import the @google/genai SDK
// here — keep all SDK calls inside `"use server"` files (Next.js Server Actions).

export type AnalysisProvider = "gemini" | "custom";

export type AnalysisStatus = "ok" | "not_a_leaf" | "out_of_scope";

export interface TopPrediction {
  className: string;
  label: string;
  probability: number; // 0-100
}

export interface AnalysisResult {
  // Routing/provenance
  status: AnalysisStatus;
  provider: AnalysisProvider;

  // Core
  isHealthy: boolean;
  diseaseName: string;
  rawClassName?: string | null;
  confidence: number; // 0-100

  // Rich fields (filled when status === "ok")
  description?: string;
  symptoms?: string[];
  treatment?: string[];
  prevention?: string[];
  severity?: number; // 0-100
  progression?: { stage: string; timeline: string }[];
  environmentalFactors?: {
    temperature: string;
    humidity: string;
    sunlight: string;
    watering: string;
  };
  topPredictions?: TopPrediction[];

  // Rejection metadata (filled when status !== "ok")
  message?: string;
  leafProbability?: number; // 0-100
  entropy?: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface BackendHealth {
  reachable: boolean;
  modelLoaded: boolean;
  leafModelLoaded: boolean;
  classesCount: number;
  diseaseInfoLoaded: boolean;
  thresholds?: { leaf: number; confidence: number; entropy: number };
  error?: string;
}

export interface SupportedClasses {
  totalClasses: number;
  speciesCount: number;
  bySpecies: Record<string, string[]>;
  limitations: string[];
  fallbackAdvice: string;
}
