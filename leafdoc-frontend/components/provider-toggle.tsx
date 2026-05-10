"use client";

import { useEffect, useState, useCallback } from "react";
import type { AnalysisProvider, BackendHealth } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Sparkles, Cpu, RefreshCw } from "lucide-react";
import { cn } from "@/lib/utils";
import { checkBackendHealthAction } from "@/app/actions/health";

interface Props {
  provider: AnalysisProvider;
  onChange: (p: AnalysisProvider) => void;
  disabled?: boolean;
}

export function ProviderToggle({ provider, onChange, disabled }: Props) {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [checking, setChecking] = useState(false);

  const refreshHealth = useCallback(async () => {
    setChecking(true);
    try {
      const res = await checkBackendHealthAction();
      setHealth(res);
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    if (provider === "custom") {
      void refreshHealth();
    }
  }, [provider, refreshHealth]);

  const customReady = health?.reachable && health.modelLoaded && health.classesCount > 0;
  const showWarning = provider === "custom" && health && !customReady;

  return (
    <Card className="p-3">
      <div className="grid grid-cols-2 gap-2">
        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange("openrouter")}
          className={cn(
            "flex flex-col items-start gap-1 rounded-md border p-3 text-left transition-colors",
            "hover:bg-muted/50 disabled:opacity-50 disabled:cursor-not-allowed",
            provider === "openrouter"
              ? "border-primary bg-primary/5"
              : "border-border"
          )}
        >
          <div className="flex items-center gap-2 font-medium">
            <Sparkles className="h-4 w-4" />
            OpenRouter
          </div>
          <div className="text-xs text-muted-foreground">
            Free AI. Recognizes most plants & conditions.
          </div>
        </button>

        <button
          type="button"
          disabled={disabled}
          onClick={() => onChange("custom")}
          className={cn(
            "flex flex-col items-start gap-1 rounded-md border p-3 text-left transition-colors",
            "hover:bg-muted/50 disabled:opacity-50 disabled:cursor-not-allowed",
            provider === "custom"
              ? "border-primary bg-primary/5"
              : "border-border"
          )}
        >
          <div className="flex items-center gap-2 font-medium">
            <Cpu className="h-4 w-4" />
            Custom Model
          </div>
          <div className="text-xs text-muted-foreground flex items-center gap-2">
            {health && health.reachable && health.modelLoaded ? (
              <span className="flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-green-500" />
                Ready ({health.classesCount} classes)
              </span>
            ) : provider === "custom" && checking ? (
              <span className="flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-yellow-500 animate-pulse" />
                Checking...
              </span>
            ) : provider === "custom" && health && !health.reachable ? (
              <span className="flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
                Backend unreachable
              </span>
            ) : provider === "custom" && health && !health.modelLoaded ? (
              <span className="flex items-center gap-1">
                <span className="h-1.5 w-1.5 rounded-full bg-orange-500" />
                Model not loaded
              </span>
            ) : (
              <span>FastAPI · 38 classes</span>
            )}
          </div>
        </button>
      </div>

      {showWarning && (
        <div className="mt-3 flex items-center justify-between gap-2 rounded-md border border-orange-200 bg-orange-50 px-3 py-2 text-xs text-orange-900 dark:border-orange-900/50 dark:bg-orange-950/30 dark:text-orange-200">
          <div>
            {!health?.reachable
              ? `Could not reach the backend. Start it with \`uvicorn main:app --reload\` in leafdoc-backend.`
              : !health.modelLoaded
                ? "Backend is up but the .keras model isn't loaded. See colab/README.md."
                : "Backend not fully ready."}
          </div>
          <Button
            type="button"
            size="sm"
            variant="ghost"
            className="h-7 px-2"
            onClick={() => void refreshHealth()}
            disabled={checking}
          >
            <RefreshCw className={cn("h-3 w-3", checking && "animate-spin")} />
          </Button>
        </div>
      )}
    </Card>
  );
}
