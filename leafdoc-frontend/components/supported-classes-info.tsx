"use client";

import { useEffect, useState } from "react";
import { Sheet, SheetContent, SheetTrigger, SheetHeader, SheetTitle, SheetDescription } from "@/components/ui/sheet";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Info, Leaf, AlertTriangle } from "lucide-react";
import type { SupportedClasses } from "@/lib/types";
import { getSupportedClassesAction } from "@/app/actions/supported-classes";

export function SupportedClassesInfo() {
  const [data, setData] = useState<SupportedClasses | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open || data) return;
    let cancelled = false;
    void (async () => {
      const res = await getSupportedClassesAction();
      if (cancelled) return;
      if (res.success) setData(res.data);
      else setError(res.error);
    })();
    return () => {
      cancelled = true;
    };
  }, [open, data]);

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm" className="gap-1 text-xs text-muted-foreground">
          <Info className="h-3 w-3" />
          What can the custom model detect?
        </Button>
      </SheetTrigger>
      <SheetContent className="overflow-hidden">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Leaf className="h-5 w-5 text-green-600" />
            Custom Model Coverage
          </SheetTitle>
          <SheetDescription>
            What the locally-served MobileNetV3 classifier can and cannot identify.
          </SheetDescription>
        </SheetHeader>

        <ScrollArea className="h-[calc(100vh-7rem)] px-4">
          {error && (
            <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-900 dark:border-red-900/50 dark:bg-red-950/30 dark:text-red-200">
              Could not load coverage info: {error}. Is the FastAPI backend running?
            </div>
          )}

          {!data && !error && (
            <p className="text-sm text-muted-foreground">Loading coverage data…</p>
          )}

          {data && (
            <div className="space-y-6 pb-8">
              <div className="text-sm">
                <p>
                  <strong>{data.totalClasses}</strong> classes across{" "}
                  <strong>{data.speciesCount}</strong> plant species.
                </p>
              </div>

              <section>
                <h3 className="text-sm font-semibold mb-2">Supported plants & conditions</h3>
                <div className="space-y-3">
                  {Object.entries(data.bySpecies).map(([species, conditions]) => (
                    <div key={species} className="rounded-md border p-3">
                      <div className="font-medium text-sm mb-2">{species}</div>
                      <div className="flex flex-wrap gap-1.5">
                        {conditions.map((c) => (
                          <Badge
                            key={c}
                            variant={c.toLowerCase() === "healthy" ? "default" : "secondary"}
                            className="text-[11px] font-normal"
                          >
                            {c}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              <section>
                <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-orange-500" />
                  What the custom model CANNOT detect
                </h3>
                <ul className="space-y-1.5 text-sm text-muted-foreground list-disc pl-5">
                  {data.limitations.map((l) => (
                    <li key={l}>{l}</li>
                  ))}
                </ul>
              </section>

              <section className="rounded-md border bg-muted/40 p-3 text-sm">
                <strong>Tip: </strong>
                {data.fallbackAdvice}
              </section>
            </div>
          )}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  );
}
