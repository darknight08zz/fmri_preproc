"use client";

import { useState } from "react";
import FileUpload from "@/components/FileUpload";
import FileList from "@/components/FileList";
import ConvertedList from "@/components/ConvertedList";

const pipelineSteps = [
  { step: "01", name: "Realign", href: "/realign", desc: "Motion correction", output: "rarfunc_4D.nii", done: true, active: false },
  { step: "02", name: "Slice Timing", href: "/stc", desc: "TR slice correction", output: "srarfunc_4D.nii", done: true, active: false },
  { step: "03", name: "Coregistration", href: "/coreg", desc: "func → struct align", output: "mT1w.nii", done: true, active: false },
  { step: "04", name: "Segmentation", href: "/segment", desc: "GM / WM / CSF masks", output: "c1/c2/c3*.nii", done: true, active: false },
  { step: "05", name: "Normalisation", href: "/normalise", desc: "MNI space warp", output: "wrarfunc_4D.nii", done: false, active: true },
  { step: "06", name: "Smoothing", href: "/smooth", desc: "Gaussian FWHM kernel", output: "swrarfunc_4D.nii", done: false, active: false },
];

const doneCount = pipelineSteps.filter((s) => s.done).length;

export default function Home() {
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [conversionTrigger, setConversionTrigger] = useState(0);

  return (
    <div className="min-h-[90vh] text-white">

      {/* ────────────────────────────── HERO ───────────────────────────── */}
      <section className="relative overflow-hidden border-b border-white/[0.04] py-12 px-6">
        <div className="grid-bg" />
        <div className="glow-orb" />

        <div className="relative max-w-[1400px] mx-auto">
          {/* Badge */}
          <div className="inline-flex items-center gap-[7px] px-3 py-1 rounded-full border border-[#10b9812e] bg-[#10b9810d] mb-7">
            <div className="badge-dot" />
            <span className="text-[#34d399] text-[12px] tracking-[0.15em] uppercase">
              ADNI · SPM12 · rs-fMRI
            </span>
          </div>

          <h1 className="text-[72px] font-black tracking-[-0.05em] leading-[0.92] mb-5">
            fMRI<br />
            <span className="text-[#34d399]">Analytics</span>
          </h1>

          <p className="text-white/35 text-[18px] leading-[1.7] font-light font-sans max-w-[560px] mb-8">
            Upload DICOM files, convert to NIfTI, and run a verified SPM preprocessing
            pipeline — from raw acquisition to analysis-ready{" "}

          </p>

          {/* Stats */}
          <div className="flex flex-wrap gap-8">
            {[
              { val: "6", label: "Pipeline Steps" },
              { val: "MNI", label: "Output Space" },
              { val: "BOLD", label: "Signal Type" },
              { val: "4D", label: "NIfTI Format" },
            ].map((s) => (
              <div key={s.label} className="flex flex-col gap-1">
                <span className="text-white font-black text-[22px] leading-none">{s.val}</span>
                <span className="text-white/20 text-[11px] tracking-[0.18em] uppercase">{s.label}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ──────────────────── UPLOAD / FILE MANAGEMENT ──────────────────── */}
      <section className="max-w-[1400px] mx-auto px-6 py-10">
        {/* Section header */}
        <div className="flex items-center gap-3 mb-7">
          <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">Step 00</span>
          <div className="flex-1 h-px bg-white/[0.04]" />
          <span className="text-white/25 text-[12px]">DICOM → NIfTI Conversion</span>
        </div>

        <div className="grid md:grid-cols-2 gap-5 items-start">
          <FileUpload onUploadSuccess={() => setRefreshTrigger((p) => p + 1)} />
          <div className="space-y-5">
            <FileList
              refreshTrigger={refreshTrigger}
              onConversionSuccess={() => setConversionTrigger((p) => p + 1)}
            />
            <ConvertedList refreshTrigger={conversionTrigger} />
          </div>
        </div>
      </section>

      {/* ───────────────────────── PIPELINE STEPS ───────────────────────── */}
      <section className="border-t border-white/[0.04]">
        <div className="max-w-[1400px] mx-auto px-6 py-12">
          {/* Section header */}
          <div className="flex items-center justify-between mb-12">
            <span className="text-white/20 text-[10px] tracking-[0.3em] uppercase font-bold">Pipeline</span>
            <span className="text-white/20 text-[12px] font-mono tracking-widest">Steps 01 → 06</span>
          </div>

          {/* ── Desktop horizontal steps ── */}
          <div className="pipeline-steps relative">
            {/* Track line */}
            <div className="pipeline-track h-[1px] bg-white/[0.04] absolute top-[25px] left-0 right-0 z-0">
              <div
                className="pipeline-progress h-full bg-[#10b9814d] shadow-[0_0_10px_#10b98133] transition-all duration-700"
                style={{ width: `${(doneCount / (pipelineSteps.length - 1)) * 100}%` }}
              />
            </div>

            <div className="relative z-10 flex justify-between items-start">
              {pipelineSteps.map((s) => (
                <a
                  key={s.step}
                  href={s.href}
                  className="group flex flex-col items-center text-center px-1 no-underline flex-1"
                >
                  {/* Node */}
                  <div
                    className={`
                      step-node
                      ${s.active ? "active border-[#34d39980] bg-[#34d39908]" : s.done ? "done" : ""}
                    `}
                  >
                    {s.done && !s.active ? (
                      <span className="checkmark text-emerald-500">✓</span>
                    ) : s.active ? (
                      <div className="pulse-dot" />
                    ) : (
                      <span className="opacity-40">{s.step}</span>
                    )}
                  </div>

                  <span
                    className={`
                      text-[11px] font-bold tracking-[0.02em] mb-1 transition-colors
                      ${s.active ? "text-[#34d399]" : s.done ? "text-white/60" : "text-white/20 group-hover:text-white/40"}
                    `}
                  >
                    {s.name}
                  </span>

                  <span className="text-[12px] text-white/10 leading-snug mb-3 uppercase tracking-tight group-hover:text-white/25 transition-colors">
                    {s.desc}
                  </span>

                  <code
                    className={`
                      text-[8px] px-2 py-[3px] rounded bg-black/40 border font-mono truncate max-w-[90%] transition-colors
                      ${s.active
                        ? "border-[#10b9814d] text-[#34d399a0]"
                        : "border-white/[0.04] text-white/10 group-hover:text-white/20"
                      }
                    `}
                  >
                    {s.output}
                  </code>
                </a>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* ──────────────────────── BOTTOM DECORATION ─────────────────────── */}
      <div className="flex items-center justify-center gap-2 py-6">
        <div className="h-px bg-[#10b9814d] opacity-80" style={{ width: 40 }} />
        <div className="h-px bg-[#10b9814d] opacity-50" style={{ width: 28 }} />
        <div className="h-px bg-[#10b9814d] opacity-30" style={{ width: 16 }} />
        <span className="text-white/10 text-[12px] tracking-[0.2em] uppercase px-3">Analysis Ready</span>
        <div className="h-px bg-[#10b9814d] opacity-30" style={{ width: 16 }} />
        <div className="h-px bg-[#10b9814d] opacity-50" style={{ width: 28 }} />
        <div className="h-px bg-[#10b9814d] opacity-80" style={{ width: 40 }} />
      </div>

    </div>
  );
}