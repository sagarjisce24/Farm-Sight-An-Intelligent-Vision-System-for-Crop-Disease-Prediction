"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { AnalysisResult } from "@/lib/types";

interface AnalysisChartsProps {
  data: AnalysisResult;
}

export function AnalysisCharts({ data }: AnalysisChartsProps) {
  const severity = data.severity ?? 0;
  const progression = data.progression ?? [];
  const env = data.environmentalFactors;

  // Transform progression data for LineChart
  const progressionData = progression.map((item, index) => ({
    name: item.timeline,
    severity:
      index === 0 ? severity * 0.5 : index === 1 ? severity : Math.min(100, severity * 1.5),
    stage: item.stage,
  }));

  // Transform environmental data for RadarChart
  // We need to parse the qualitative strings into quantitative values for the chart
  // This is a rough estimation for visualization purposes
  const getEnvScore = (val?: string) => {
    if (!val) return 60;
    // Simple heuristic: if it mentions 'high' -> 80, 'low' -> 20, 'medium'/'moderate' -> 50
    const lower = val.toLowerCase();
    if (lower.includes("high")) return 80;
    if (lower.includes("low")) return 20;
    if (lower.includes("warm") || lower.includes("cool")) return 50;
    return 60; // Default
  };

  const envData = [
    { subject: "Temperature", A: getEnvScore(env?.temperature), fullMark: 100 },
    { subject: "Humidity", A: getEnvScore(env?.humidity), fullMark: 100 },
    { subject: "Sunlight", A: getEnvScore(env?.sunlight), fullMark: 100 },
    { subject: "Watering", A: getEnvScore(env?.watering), fullMark: 100 },
  ];

  if (data.isHealthy) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Health Metrics</CardTitle>
          <CardDescription>Plant vitality and growth indicators</CardDescription>
        </CardHeader>
        <CardContent className="h-[300px] flex items-center justify-center">
            <p className="text-muted-foreground">Your plant is healthy! No disease analytics required.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
      <Card>
        <CardHeader>
          <CardTitle>Disease Progression</CardTitle>
          <CardDescription>Estimated severity over time without treatment</CardDescription>
        </CardHeader>
        <CardContent className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={progressionData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="severity" stroke="#ef4444" strokeWidth={2} name="Severity %" activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Environmental Analysis</CardTitle>
          <CardDescription>Factor impact assessment</CardDescription>
        </CardHeader>
        <CardContent className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={envData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="subject" />
              <PolarRadiusAxis angle={30} domain={[0, 100]} />
              <Radar name="Impact Factor" dataKey="A" stroke="#22c55e" fill="#22c55e" fillOpacity={0.6} />
              <Tooltip />
            </RadarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
