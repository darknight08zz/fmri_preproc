"use client";

import Link from "next/link";

export default function Header() {
  return (
    <header className="w-full bg-transparent border-b border-white/[0.04] px-6">
      <div className="max-w-[1400px] mx-auto h-16 flex items-center justify-between">

        {/* ── Brand ── */}
        <Link href="/" className="flex items-center gap-[10px] no-underline">
          <div className="logo-icon">
            <svg width="16" height="16" viewBox="0 0 18 18" fill="none" className="text-emerald-400">
              <circle cx="9" cy="9" r="6.5" stroke="currentColor" strokeWidth="0.8" />
              <circle cx="9" cy="9" r="3.5" stroke="currentColor" strokeWidth="0.5" strokeDasharray="2 1" />
              <line x1="5" y1="9" x2="6.5" y2="9" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
              <line x1="11.5" y1="9" x2="13" y2="9" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
              <line x1="9" y1="5" x2="9" y2="6.5" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
              <line x1="9" y1="11.5" x2="9" y2="13" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
              <circle cx="9" cy="9" r="1.2" fill="currentColor" />
            </svg>
          </div>

          <div className="flex flex-col gap-[2px]">
            <div className="text-white font-black text-base tracking-tighter leading-none">
              fMRI<span className="text-emerald-400">Preproc</span>
            </div>
            <div className="text-white/20 text-[11px] tracking-[0.18em] uppercase font-mono">
              SPM Pipeline
            </div>
          </div>
        </Link>

        {/* ── Live status ── */}
        <div className="flex items-center gap-1.5 text-[11px] text-white/25">
          <div className="dot-ping" />
          Pipeline Active
        </div>

      </div>
    </header>
  );
}