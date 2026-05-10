"use client";

import type { AnalysisResult } from "@/lib/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { X, CheckCircle, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ComparisonViewProps {
  items: AnalysisResult[];
  onRemove: (index: number) => void;
  onClear: () => void;
}

export function ComparisonView({ items, onRemove, onClear }: ComparisonViewProps) {
  if (items.length === 0) return null;

  return (
    <div className="space-y-4 animate-in slide-in-from-bottom-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">Comparison Mode</h2>
        <Button variant="ghost" onClick={onClear} className="text-destructive hover:text-destructive">
          Clear Comparison
        </Button>
      </div>

      <ScrollArea className="w-full whitespace-nowrap rounded-md border">
        <div className="flex w-max space-x-4 p-4">
          {items.map((result, i) => (
            <Card key={i} className="w-[350px] shrink-0 border-2 relative">
              <Button
                variant="ghost"
                size="icon"
                className="absolute top-2 right-2 h-6 w-6 rounded-full bg-background/80 hover:bg-destructive hover:text-destructive-foreground z-10 shadow-sm"
                onClick={() => onRemove(i)}
              >
                <X className="h-4 w-4" />
              </Button>
              <CardHeader>
                <div className="flex items-center justify-between mb-2">
                  <Badge variant={result.isHealthy ? "default" : "destructive"}>
                    {result.isHealthy ? "Healthy" : "Infected"}
                  </Badge>
                  <span className="text-sm font-mono text-muted-foreground">{result.confidence}% Conf.</span>
                </div>
                <CardTitle className="flex items-center gap-2 text-xl truncate" title={result.diseaseName}>
                  {result.isHealthy ? (
                    <CheckCircle className="h-5 w-5 text-green-500 shrink-0" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-destructive shrink-0" />
                  )}
                  {result.diseaseName}
                </CardTitle>
                <CardDescription className="line-clamp-2 h-[40px] whitespace-normal">
                  {result.description}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                    <p className="text-sm font-medium">Severity</p>
                    <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
                        <div 
                            className="h-full bg-primary transition-all" 
                            style={{ width: `${result.severity ?? 0}%`, backgroundColor: (result.severity ?? 0) > 50 ? 'var(--destructive)' : 'var(--primary)' }}
                        />
                    </div>
                </div>

                <div className="space-y-1">
                    <p className="text-sm font-medium">Environmental Factors</p>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="bg-muted p-2 rounded">
                            <span className="block text-muted-foreground">Temp</span>
                            <span className="font-medium truncate">{result.environmentalFactors?.temperature ?? "—"}</span>
                        </div>
                         <div className="bg-muted p-2 rounded">
                            <span className="block text-muted-foreground">Humidity</span>
                            <span className="font-medium truncate">{result.environmentalFactors?.humidity ?? "—"}</span>
                        </div>
                    </div>
                </div>

                <div className="space-y-1">
                     <p className="text-sm font-medium">Top Symptom</p>
                     <p className="text-sm text-muted-foreground whitespace-normal line-clamp-2">
                        {result.symptoms?.[0] ?? "No symptoms recorded."}
                     </p>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  );
}
