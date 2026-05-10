"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Leaf, History } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

export function Header() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-14 items-center justify-between mx-auto px-4">
        <div className="flex items-center gap-6">
          <Link href="/" className="flex items-center gap-2 font-bold text-xl text-primary">
            <Leaf className="h-6 w-6" />
            <span>LeafDoc</span>
          </Link>
          <nav className="hidden md:flex items-center gap-4 text-sm font-medium">
            <Link
              href="/"
              className={cn(
                "transition-colors hover:text-foreground/80",
                pathname === "/" ? "text-foreground" : "text-foreground/60"
              )}
            >
              Detector
            </Link>
            <Link
              href="/recommendations"
              className={cn(
                "transition-colors hover:text-foreground/80",
                pathname === "/recommendations" ? "text-foreground" : "text-foreground/60"
              )}
            >
              Recommendations
            </Link>
          </nav>
        </div>

        {/* This button logic needs to be lifted up or handled differently since History is specific to the home page analysis */}
        {/* For now, we will only show the History trigger if we are on the home page, but the state lives in Page... */}
        {/* Actually, the layout wrapping might make it hard to trigger the Sheet inside Page. */}
        {/* A better approach for now: Leave the History button IN the Page header area, and this component just handles main NAV. */}
        {/* OR, we move history context to a global provider. But for simplicity, let's keep History local to the detector page for now and effectively hide it here or duplicate the header logic? */}
        {/* Re-reading requirements: "Create a header component with two distinct pages". The existing code had the header INSIDE page.tsx. */}
        {/* I should REMOVE the header from page.tsx and put it in layout.tsx. */}
        {/* Issue: The History Sheet is coupled to the 'history' state in page.tsx. */}
        {/* Solution: I'll keep the standard nav items here. The page-specific actions (like History) can explicitly be rendered in the Page's sub-header or I can ignore it for the moment to proceed with requirements. */}
        {/* Let's render a simple placeholder for right-side actions if needed, but for now just navigation. */}
      </div>
    </header>
  );
}
