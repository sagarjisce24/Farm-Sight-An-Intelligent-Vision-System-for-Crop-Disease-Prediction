"use client";

import { useState } from "react";
import { useForm } from "react-hook-form";
import { getRecommendationsAction, type RecommendationResult } from "@/app/actions/recommend";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, MapPin, CloudSun, Leaf, BookOpen, CheckCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";

// Mock Weather Service (Client-side for now, could be server-side)
// In a real app, we'd use an API key for OpenWeatherMap here or in a server action.
const fetchWeather = async (lat: number, lon: number) => {
    // Simulating API call delay
    await new Promise(resolve => setTimeout(resolve, 800));
    return {
        temp: "24°C",
        condition: "Partly Cloudy",
        humidity: "65%",
        forecast: "Rain expected next week"
    };
};

export default function RecommendationsPage() {
    const [status, setStatus] = useState<"IDLE" | "LOADING_LOC" | "LOADING_WEATHER" | "GENERATING" | "SUCCESS" | "ERROR">("IDLE");
    const [locationName, setLocationName] = useState("");
    const [weather, setWeather] = useState<any>(null);
    const [recommendations, setRecommendations] = useState<RecommendationResult | null>(null);
    const { register, handleSubmit } = useForm<{ city: string }>();

    const handleUseMyLocation = () => {
        setStatus("LOADING_LOC");
        if (!navigator.geolocation) {
            alert("Geolocation is not supported by your browser");
            setStatus("IDLE");
            return;
        }

        navigator.geolocation.getCurrentPosition(async (position) => {
            const { latitude, longitude } = position.coords;
            setLocationName(`Lat: ${latitude.toFixed(2)}, Lon: ${longitude.toFixed(2)}`);
            await loadContextAndRecommend(latitude, longitude, `Current Location (${latitude.toFixed(2)}, ${longitude.toFixed(2)})`);
        }, () => {
            alert("Unable to retrieve your location");
            setStatus("IDLE");
        });
    };

    const onSubmitCity = async (data: { city: string }) => {
        if (!data.city) return;
        // For city input, we'd typically geocode it first to get precise weather.
        // For this demo, we'll pass the city name directly to our mock/AI.
        // We'll mock lat/lon or just pass string to AI.
        await loadContextAndRecommend(0, 0, data.city); 
    };

    const loadContextAndRecommend = async (lat: number, lon: number, locName: string) => {
        setStatus("LOADING_WEATHER");
        setLocationName(locName);
        try {
            const weatherData = await fetchWeather(lat, lon);
            setWeather(weatherData);

            setStatus("GENERATING");
            const result = await getRecommendationsAction(locName, weatherData);

            if (result.success && result.data) {
                setRecommendations(result.data);
                setStatus("SUCCESS");
            } else {
                throw new Error(result.error || "Failed to generate");
            }
        } catch (error) {
            console.error(error);
            setStatus("ERROR");
        }
    };

    return (
        <div className="container mx-auto px-4 py-8 max-w-5xl space-y-8">
            <div className="text-center space-y-4">
                <h1 className="text-3xl font-bold tracking-tight sm:text-4xl text-primary flex items-center justify-center gap-3">
                    <Leaf className="h-8 w-8" />
                    Plant Recommendations
                </h1>
                <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
                    Get personalized crop and plant suggestions based on your local climate, soil conditions, and seasonal forecasts.
                </p>
            </div>

            <Card className="max-w-xl mx-auto">
                <CardHeader>
                    <CardTitle>Set Location</CardTitle>
                    <CardDescription>We need your location to analyze local weather patterns.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="flex gap-2">
                         <Button variant="outline" className="w-full" onClick={handleUseMyLocation} disabled={status !== "IDLE" && status !== "ERROR" && status !== "SUCCESS"}>
                            <MapPin className="mr-2 h-4 w-4" />
                            Use My Location
                         </Button>
                    </div>
                    <div className="relative">
                        <div className="absolute inset-0 flex items-center">
                            <span className="w-full border-t" />
                        </div>
                        <div className="relative flex justify-center text-xs uppercase">
                            <span className="bg-background px-2 text-muted-foreground">Or enter manually</span>
                        </div>
                    </div>
                    <form onSubmit={handleSubmit(onSubmitCity)} className="flex gap-2">
                        <Input placeholder="Enter city name..." {...register("city")} disabled={status !== "IDLE" && status !== "ERROR" && status !== "SUCCESS"} />
                        <Button type="submit" disabled={status !== "IDLE" && status !== "ERROR" && status !== "SUCCESS"}>Go</Button>
                    </form>
                </CardContent>
            </Card>

            {status !== "IDLE" && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                     {/* Context Display */}
                     {(status === "LOADING_WEATHER" || status === "GENERATING" || status === "SUCCESS") && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium text-muted-foreground">Location</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    <div className="flex items-center gap-2">
                                        <MapPin className="h-5 w-5 text-primary" />
                                        <span className="font-semibold text-lg">{locationName}</span>
                                    </div>
                                </CardContent>
                            </Card>
                            <Card>
                                <CardHeader className="pb-2">
                                    <CardTitle className="text-sm font-medium text-muted-foreground">Current Weather</CardTitle>
                                </CardHeader>
                                <CardContent>
                                    {weather ? (
                                        <div className="flex items-center gap-4">
                                            <CloudSun className="h-8 w-8 text-yellow-500" />
                                            <div>
                                                <p className="font-semibold text-lg">{weather.temp}, {weather.condition}</p>
                                                <p className="text-sm text-muted-foreground">Humidity: {weather.humidity}</p>
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="flex items-center gap-2 text-muted-foreground">
                                            <Loader2 className="h-4 w-4 animate-spin" />
                                            Fetching weather info...
                                        </div>
                                    )}
                                </CardContent>
                            </Card>
                        </div>
                     )}

                     {status === "GENERATING" && (
                         <div className="flex flex-col items-center justify-center py-12 space-y-4">
                             <Loader2 className="h-10 w-10 animate-spin text-primary" />
                             <p className="text-lg font-medium animate-pulse">Consulting AI Agronomist...</p>
                         </div>
                     )}

                     {status === "SUCCESS" && recommendations && (
                        <div className="space-y-6">
                            <h2 className="text-2xl font-bold border-b pb-2">AI Recommendations</h2>
                            
                            <Tabs defaultValue="crops" className="w-full">
                                <TabsList>
                                    <TabsTrigger value="crops">Suggested Crops</TabsTrigger>
                                    <TabsTrigger value="guide">Cultivation Guide</TabsTrigger>
                                </TabsList>
                                
                                <TabsContent value="crops" className="mt-4 space-y-4">
                                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                        {recommendations.crops.map((crop, i) => (
                                            <Card key={i} className="hover:shadow-md transition-shadow">
                                                <CardHeader>
                                                    <div className="flex justify-between items-start">
                                                        <CardTitle>{crop.name}</CardTitle>
                                                        <Badge variant={crop.type === "Short-term" ? "secondary" : "default"}>{crop.type}</Badge>
                                                    </div>
                                                    <CardDescription>{crop.cultivationTime} to harvest</CardDescription>
                                                </CardHeader>
                                                <CardContent className="space-y-3">
                                                    <div className="text-sm">
                                                        <span className="font-semibold">Suitability:</span>
                                                        <div className="w-full bg-secondary h-2 rounded-full mt-1">
                                                            <div className="bg-green-500 h-full rounded-full" style={{ width: `${crop.suitabilityScore}%` }} />
                                                        </div>
                                                    </div>
                                                    <p className="text-sm text-muted-foreground">{crop.description}</p>
                                                    <div>
                                                        <p className="text-xs font-semibold mb-1 uppercase text-muted-foreground">Benefits</p>
                                                        <div className="flex flex-wrap gap-1">
                                                            {crop.benefits.map((b, j) => (
                                                                <Badge key={j} variant="outline" className="text-xs">{b}</Badge>
                                                            ))}
                                                        </div>
                                                    </div>
                                                </CardContent>
                                            </Card>
                                        ))}
                                    </div>
                                </TabsContent>

                                <TabsContent value="guide" className="mt-4">
                                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                        <Card className="md:col-span-2">
                                            <CardHeader>
                                                <CardTitle className="flex items-center gap-2">
                                                    <BookOpen className="h-5 w-5" />
                                                    Seasonal Guide
                                                </CardTitle>
                                            </CardHeader>
                                            <CardContent>
                                                 <ScrollArea className="h-[400px]">
                                                    <div className="prose dark:prose-invert max-w-none">
                                                        <p className="whitespace-pre-line leading-relaxed">{recommendations.cultivationGuide}</p>
                                                    </div>
                                                 </ScrollArea>
                                            </CardContent>
                                        </Card>
                                        <Card>
                                            <CardHeader>
                                                <CardTitle className="flex items-center gap-2">
                                                    <CheckCircle className="h-5 w-5" />
                                                    Best Practices
                                                </CardTitle>
                                            </CardHeader>
                                            <CardContent>
                                                <ul className="space-y-3">
                                                    {recommendations.bestPractices.map((practice, i) => (
                                                        <li key={i} className="flex gap-2 text-sm">
                                                            <span className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5 shrink-0" />
                                                            <span>{practice}</span>
                                                        </li>
                                                    ))}
                                                </ul>
                                            </CardContent>
                                        </Card>
                                    </div>
                                </TabsContent>
                            </Tabs>
                        </div>
                     )}
                </div>
            )}
        </div>
    );
}
