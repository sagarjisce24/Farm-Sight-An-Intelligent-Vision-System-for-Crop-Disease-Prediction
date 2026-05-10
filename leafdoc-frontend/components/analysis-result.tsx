"use client";

import type { AnalysisResult } from "@/lib/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  AlertCircle,
  CheckCircle,
  Leaf,
  Droplets,
  Sun,
  Thermometer,
  HelpCircle,
  ImageOff,
  Sparkles,
  Cpu,
} from "lucide-react";
import { AnalysisCharts } from "./analysis-charts";
import { ScrollArea } from "@/components/ui/scroll-area";
import { DiseaseChat } from "./disease-chat";

interface Props {
  result: AnalysisResult;
  onSwitchToOpenRouter?: () => void;
}

export function AnalysisResultDisplay({ result, onSwitchToOpenRouter }: Props) {
  if (result.status === "not_a_leaf") {
    return <NotALeafCard result={result} onSwitchToOpenRouter={onSwitchToOpenRouter} />;
  }
  if (result.status === "out_of_scope") {
    return <OutOfScopeCard result={result} onSwitchToOpenRouter={onSwitchToOpenRouter} />;
  }
  return <SuccessResult result={result} />;
}

// -----------------------------------------------------------------------------
// not_a_leaf
// -----------------------------------------------------------------------------

function NotALeafCard({
  result,
  onSwitchToOpenRouter,
}: {
  result: AnalysisResult;
  onSwitchToOpenRouter?: () => void;
}) {
  return (
    <Card className="border-l-4 border-l-orange-500 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <ImageOff className="h-5 w-5 text-orange-500" />
          Not a leaf
        </CardTitle>
        <CardDescription>The leaf-detection gate rejected this image.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{result.message}</p>
        {typeof result.leafProbability === "number" && (
          <div className="text-xs text-muted-foreground">
            Leaf probability: <span className="font-medium">{result.leafProbability}%</span>
          </div>
        )}
        {onSwitchToOpenRouter && (
          <Button onClick={onSwitchToOpenRouter} variant="default" className="gap-2">
            <Sparkles className="h-4 w-4" />
            Re-analyze with OpenRouter
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// -----------------------------------------------------------------------------
// out_of_scope
// -----------------------------------------------------------------------------

function OutOfScopeCard({
  result,
  onSwitchToOpenRouter,
}: {
  result: AnalysisResult;
  onSwitchToOpenRouter?: () => void;
}) {
  return (
    <Card className="border-l-4 border-l-yellow-500 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-xl">
          <HelpCircle className="h-5 w-5 text-yellow-600" />
          Could not confidently identify
        </CardTitle>
        <CardDescription>
          Looks like a leaf, but the model couldn't match it to one of its 38 supported classes.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-sm text-muted-foreground">{result.message}</p>
        <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
          <span>
            Top confidence: <span className="font-medium">{result.confidence}%</span>
          </span>
          {typeof result.entropy === "number" && (
            <span>
              Entropy: <span className="font-medium">{result.entropy.toFixed(2)}</span>
            </span>
          )}
          {typeof result.leafProbability === "number" && (
            <span>
              Leaf probability: <span className="font-medium">{result.leafProbability}%</span>
            </span>
          )}
        </div>

        {result.topPredictions && result.topPredictions.length > 0 && (
          <div className="space-y-1.5">
            <p className="text-xs font-medium text-muted-foreground">Closest guesses:</p>
            <ul className="text-xs space-y-1">
              {result.topPredictions.map((p) => (
                <li key={p.className} className="flex justify-between">
                  <span>{p.label}</span>
                  <span className="text-muted-foreground">{p.probability.toFixed(1)}%</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {onSwitchToOpenRouter && (
          <Button onClick={onSwitchToOpenRouter} variant="default" className="gap-2">
            <Sparkles className="h-4 w-4" />
            Re-analyze with OpenRouter
          </Button>
        )}
      </CardContent>
    </Card>
  );
}

// -----------------------------------------------------------------------------
// ok (full result)
// -----------------------------------------------------------------------------

function SuccessResult({ result }: { result: AnalysisResult }) {
  const symptoms = result.symptoms ?? [];
  const treatment = result.treatment ?? [];
  const prevention = result.prevention ?? [];
  const env = result.environmentalFactors;

  return (
    <div className="w-full space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <Card className="border-l-4 border-l-primary">
        <CardHeader>
          <div className="flex items-start justify-between gap-3">
            <div className="space-y-1">
              <CardTitle className="text-2xl flex items-center gap-2">
                {result.isHealthy ? (
                  <CheckCircle className="h-6 w-6 text-green-500" />
                ) : (
                  <AlertCircle className="h-6 w-6 text-destructive" />
                )}
                {result.diseaseName}
              </CardTitle>
              <CardDescription className="flex items-center gap-3 flex-wrap">
                <span>Confidence: {result.confidence}%</span>
                <Badge variant="outline" className="gap-1 text-[10px] font-normal">
                  {result.provider === "openrouter" ? (
                    <>
                      <Sparkles className="h-3 w-3" /> OpenRouter
                    </>
                  ) : (
                    <>
                      <Cpu className="h-3 w-3" /> Custom Model
                    </>
                  )}
                </Badge>
              </CardDescription>
            </div>
            <Badge variant={result.isHealthy ? "default" : "destructive"}>
              {result.isHealthy ? "Healthy" : "Infected"}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Severity</span>
              <span>{Math.round(result.severity ?? 0)}%</span>
            </div>
            <Progress value={result.severity ?? 0} className="h-2" />
            {result.description && (
              <p className="mt-4 text-muted-foreground">{result.description}</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="overview" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="treatment">Treatment</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Symptoms</CardTitle>
              </CardHeader>
              <CardContent>
                {symptoms.length > 0 ? (
                  <ul className="list-disc pl-5 space-y-1 text-muted-foreground">
                    {symptoms.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ul>
                ) : (
                  <p className="text-muted-foreground text-sm">No symptoms reported.</p>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Environment</CardTitle>
              </CardHeader>
              <CardContent>
                {env ? (
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <Thermometer className="h-4 w-4 text-orange-500" />
                      <span className="text-sm">{env.temperature}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <Droplets className="h-4 w-4 text-blue-500" />
                      <span className="text-sm">{env.humidity}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <Sun className="h-4 w-4 text-yellow-500" />
                      <span className="text-sm">{env.sunlight}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <Leaf className="h-4 w-4 text-green-500" />
                      <span className="text-sm">{env.watering}</span>
                    </div>
                  </div>
                ) : (
                  <p className="text-muted-foreground text-sm">No environment data.</p>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="treatment" className="space-y-4 mt-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Treatment Plan</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[200px]">
                  <ol className="list-decimal pl-5 space-y-2 text-muted-foreground">
                    {treatment.map((step, i) => (
                      <li key={i}>{step}</li>
                    ))}
                  </ol>
                </ScrollArea>
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Prevention</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[200px]">
                  <ul className="list-disc pl-5 space-y-2 text-muted-foreground">
                    {prevention.map((tip, i) => (
                      <li key={i}>{tip}</li>
                    ))}
                  </ul>
                </ScrollArea>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="analytics" className="mt-4">
          <AnalysisCharts data={result} />
        </TabsContent>
      </Tabs>

      {/* Q&A is available regardless of provider — backend always handles OpenRouter calls. */}
      {!result.isHealthy && <DiseaseChat diseaseName={result.diseaseName} />}
    </div>
  );
}
