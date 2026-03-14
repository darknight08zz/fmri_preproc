"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

// ─────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────

interface ConvertedFolder {
    name: string;
    fileCount: number;
    files: string[];
    jsonFiles: string[];
}

type ExtractStatus = "idle" | "extracting" | "success" | "fallback" | "error";

interface ExtractedParams {
    TR: number;
    nSlices: number;
    nVolumes: number;
    TA: number;
    sliceOrder: number[];
    refSlice: number;
    source: string;
}

// ─────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────

const FIELD_CLASS =
    "w-full px-3 py-2 bg-[#1a1b22] text-white border border-[#2d2f3d] rounded-md " +
    "focus:outline-none focus:border-[#2563eb] font-mono text-sm transition-colors";

const AUTO_FILL_CLASS =
    "w-full px-3 py-2 bg-[#0f1a2e] text-[#60a5fa] border border-[#1e3a6e] rounded-md " +
    "font-mono text-sm";

// ─────────────────────────────────────────────
//  Component
// ─────────────────────────────────────────────

export default function STCPage() {
    const [folders, setFolders] = useState<ConvertedFolder[]>([]);
    const [loadingFiles, setLoadingFiles] = useState(true);
    const [selectedFolder, setSelectedFolder] = useState("");
    const [selectedFile, setSelectedFile] = useState("");

    const [tr, setTr] = useState("");
    const [slices, setSlices] = useState("");
    const [ta, setTa] = useState("");
    const [sliceOrder, setSliceOrder] = useState("ascending");
    const [refSlice, setRefSlice] = useState("0");

    const [extractStatus, setExtractStatus] = useState<ExtractStatus>("idle");
    const [extractMsg, setExtractMsg] = useState("");
    const [extractedRaw, setExtractedRaw] = useState<ExtractedParams | null>(null);
    const [jsonFilePath, setJsonFilePath] = useState("");
    const [autoFilled, setAutoFilled] = useState<Set<string>>(new Set());

    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<string | null>(null);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        (async () => {
            try {
                const res = await fetch("/api/converted-files");
                const data = await res.json();
                setFolders(data.folders ?? []);
            } catch {
                setFolders([]);
            } finally {
                setLoadingFiles(false);
            }
        })();
    }, []);

    const handleFolderChange = (folder: string) => {
        setSelectedFolder(folder);
        setSelectedFile("");
        setResult(null);
        setError(null);
        setStatus(null);
    };

    const resolvedFilename =
        selectedFolder && selectedFile ? `${selectedFolder}/${selectedFile}` : "";

    useEffect(() => {
        if (resolvedFilename) {
            const guessedJson = `converted/${resolvedFilename}`
                .replace(/\.nii\.gz$/, ".json")
                .replace(/\.nii$/, ".json");
            setJsonFilePath(guessedJson);
        }
    }, [resolvedFilename]);

    const deriveSliceOrderLabel = (order: number[]): string => {
        if (!order || order.length < 2) return "ascending";
        const diffs = order.slice(1).map((v, i) => v - order[i]);
        if (diffs.every((d) => d > 0)) return "ascending";
        if (diffs.every((d) => d < 0)) return "descending";
        return "interleaved";
    };

    const handleExtract = async () => {
        if (!jsonFilePath.trim()) return;
        setExtractStatus("extracting");
        setExtractMsg("Analyzing metadata...");
        setAutoFilled(new Set());

        try {
            const res = await fetch("/api/extract-stc-params", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ json_path: jsonFilePath.trim() }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error ?? "Extraction failed");

            const p = data as ExtractedParams;
            setExtractedRaw(p);
            const filled = new Set<string>();
            setTr(p.TR.toFixed(4)); filled.add("tr");
            setSlices(String(p.nSlices)); filled.add("slices");
            setTa(p.TA.toFixed(6)); filled.add("ta");
            setRefSlice(String(p.refSlice - 1)); filled.add("refSlice");
            setSliceOrder(deriveSliceOrderLabel(p.sliceOrder)); filled.add("sliceOrder");
            setAutoFilled(filled);

            const isFallback = p.source === "json_only" || !p.TR;
            setExtractStatus(isFallback ? "fallback" : "success");
            setExtractMsg(isFallback ? "Extracted from JSON (NIfTI not found)" : "Successfully extracted all parameters");
        } catch (err: any) {
            setExtractStatus("error");
            setExtractMsg(err.message ?? "Extraction failed");
        }
    };

    const handleRunSTC = async () => {
        if (!resolvedFilename) return;
        setIsLoading(true);
        setStatus("Processing...");
        setError(null);
        setResult(null);

        try {
            const response = await fetch("/api/stc", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filename: resolvedFilename,
                    tr: tr ? parseFloat(tr) : null,
                    slices: slices ? parseInt(slices) : null,
                    ta: ta ? parseFloat(ta) : null,
                    slice_order: sliceOrder,
                    ref_slice: parseInt(refSlice),
                }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Execution failed");
            setResult(data);
            setStatus("Complete");
        } catch (err: any) {
            setError(err.message);
            setStatus("Failed");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="min-h-[90vh] text-white">
            {/* ────────────────────────────── HERO ───────────────────────────── */}
            <section className="relative overflow-hidden border-b border-white/[0.04] py-10 px-6">
                <div className="grid-bg" />
                <div className="glow-orb" />
                
                <div className="relative max-w-[1400px] mx-auto">
                    <div className="inline-flex items-center gap-[7px] px-3 py-1 rounded-full border border-[#10b9812e] bg-[#10b9810d] mb-6">
                        <div className="badge-dot" />
                        <span className="text-[#34d399] text-[12px] tracking-[0.15em] uppercase">
                            Step 02 · Slice Timing
                        </span>
                    </div>

                    <h1 className="text-[52px] font-black tracking-[-0.05em] leading-[0.95] mb-4">
                        Slice <span className="text-[#34d399]">Timing</span>
                    </h1>

                    <p className="text-white/35 text-[17px] leading-[1.6] font-light font-sans max-w-[500px]">
                        Corrects for differences in slice acquisition times by shifting each 
                        slice's time-series using Fourier phase interpolation.
                    </p>
                </div>
            </section>

            {/* ────────────────────────────── CONTENT ─────────────────────────── */}
            <div className="relative max-w-[1400px] mx-auto px-6 py-10 space-y-10">
                
                {/* 01 - Volume Selection */}
                <section>
                    <div className="flex items-center gap-3 mb-6">
                        <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">02-A</span>
                        <div className="flex-1 h-px bg-white/[0.04]" />
                        <span className="text-white/25 text-[10px]">Volume Selection</span>
                    </div>

                    <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6">
                        <ImagePicker
                            badge="NIFTI"
                            badgeColor="text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
                            label="Functional Volume"
                            sublabel="4D BOLD series for temporal correction"
                            folders={folders}
                            selectedFolder={selectedFolder}
                            selectedFile={selectedFile}
                            onFolderChange={handleFolderChange}
                            onFileChange={setSelectedFile}
                            resolvedPath={resolvedFilename ? `converted/${resolvedFilename}` : ""}
                        />
                    </div>
                </section>

                {/* 02 - Auto-Extract */}
                <section>
                    <div className="flex items-center gap-3 mb-6">
                        <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">02-B</span>
                        <div className="flex-1 h-px bg-white/[0.04]" />
                        <span className="text-white/25 text-[10px]">Metadata Extraction</span>
                    </div>

                    <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6 space-y-4">
                        <div className="flex items-center justify-between">
                            <div className="flex flex-col">
                                <span className="text-[10px] text-white/40 uppercase tracking-widest">BIDS Sidecar</span>
                                <span className="text-[9px] text-white/10 font-mono italic">Extract TR/Slices from JSON</span>
                            </div>
                            {extractStatus !== "idle" && (
                                <span className={`text-[10px] font-mono ${
                                    extractStatus === "success" ? "text-emerald-400" : 
                                    extractStatus === "error" ? "text-red-400" : "text-amber-400"
                                }`}>
                                    {extractMsg}
                                </span>
                            )}
                        </div>

                        <div className="flex gap-2">
                            <input
                                value={jsonFilePath}
                                onChange={(e) => setJsonFilePath(e.target.value)}
                                className="flex-1 bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-emerald-500/40"
                                placeholder="path/to/sidecar.json"
                            />
                            <button
                                onClick={handleExtract}
                                disabled={!jsonFilePath || extractStatus === "extracting"}
                                className="px-4 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-[10px] font-bold uppercase tracking-widest rounded-lg hover:bg-emerald-500/20 transition-all disabled:opacity-30"
                            >
                                {extractStatus === "extracting" ? "..." : "Extract"}
                            </button>
                        </div>
                    </div>
                </section>

                {/* 03 - Parameters */}
                <section>
                    <div className="flex items-center gap-3 mb-6">
                        <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">02-C</span>
                        <div className="flex-1 h-px bg-white/[0.04]" />
                        <span className="text-white/25 text-[10px]">STC Parameters</span>
                    </div>

                    <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6 grid grid-cols-2 gap-x-8 gap-y-6">
                        <ParamInput
                            label="TR (seconds)"
                            value={tr}
                            onChange={setTr}
                            auto={autoFilled.has("tr")}
                        />
                        <ParamInput
                            label="Number of Slices"
                            value={slices}
                            onChange={setSlices}
                            auto={autoFilled.has("slices")}
                        />
                        <ParamInput
                            label="TA (Acquisition Time)"
                            value={ta}
                            onChange={setTa}
                            auto={autoFilled.has("ta")}
                        />
                        <ParamInput
                            label="Reference Slice"
                            value={refSlice}
                            onChange={setRefSlice}
                            auto={autoFilled.has("refSlice")}
                        />
                        <div className="col-span-2 space-y-2">
                             <label className="text-[10px] text-white/40 uppercase tracking-widest ml-1">Slice Order</label>
                             <select
                                value={sliceOrder}
                                onChange={(e) => setSliceOrder(e.target.value)}
                                className={`w-full bg-black/40 border rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-emerald-500/40 ${
                                    autoFilled.has("sliceOrder") ? "border-emerald-500/40 text-emerald-400" : "border-white/[0.05]"
                                }`}
                             >
                                <option value="ascending">Ascending (1, 2, 3...)</option>
                                <option value="descending">Descending (N, N-1...)</option>
                                <option value="interleaved">Interleaved (1, 3, 5...)</option>
                             </select>
                        </div>
                    </div>
                </section>

                {/* Run Button */}
                <button
                    onClick={handleRunSTC}
                    disabled={isLoading || !selectedFolder || !selectedFile}
                    className={`
                        w-full py-4 rounded-xl border font-bold text-xs uppercase tracking-[0.2em] transition-all
                        ${isLoading || !selectedFolder || !selectedFile
                            ? "bg-white/5 border-white/5 text-white/10 cursor-not-allowed"
                            : "bg-emerald-500/10 border-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15 hover:border-emerald-500/20 shadow-[0_4px_20px_rgba(16,185,129,0.05)]"
                        }
                    `}
                >
                    {isLoading ? "Executing Step 02..." : "Initialize Timing Correction"}
                </button>

                {/* Results/Error */}
                {(error || result) && (
                    <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6">
                        {error && (
                            <div className="text-[10px] text-red-400 font-mono whitespace-pre-wrap">{error}</div>
                        )}
                        {result && (
                            <div className="space-y-4">
                                <div className="flex items-center justify-between">
                                    <span className="text-[10px] text-emerald-400 font-bold uppercase tracking-widest">Correction Complete</span>
                                    <span className="text-[9px] text-white/20 font-mono">Status: {status}</span>
                                </div>
                                <div className="text-[10px] font-mono bg-black/40 border border-white/[0.05] rounded-lg p-3 text-white/60">
                                    <span className="text-white/20 mr-2">Output:</span>
                                    {result.output}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

function ParamInput({ label, value, onChange, auto }: any) {
    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <label className="text-[10px] text-white/40 uppercase tracking-widest ml-1">{label}</label>
                {auto && <span className="text-[8px] text-emerald-400 font-bold uppercase tracking-widest bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">Auto</span>}
            </div>
            <input
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className={`w-full bg-black/40 border rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-emerald-500/40 ${
                    auto ? "border-emerald-500/40 text-emerald-400" : "border-white/[0.05]"
                }`}
            />
        </div>
    );
}

// ── Image Picker helper (Premium Style) ──────────────────────────────
interface ImagePickerProps {
  badge: string; badgeColor: string; label: string; sublabel: string;
  folders: ConvertedFolder[]; selectedFolder: string; selectedFile: string;
  onFolderChange: (f: string) => void; onFileChange: (f: string) => void;
  resolvedPath: string;
}

function ImagePicker({
  badge, badgeColor, label, sublabel,
  folders, selectedFolder, selectedFile,
  onFolderChange, onFileChange,
  resolvedPath
}: ImagePickerProps) {
  const files = folders.find((f) => f.name === selectedFolder)?.files ?? [];
  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className={`text-[9px] px-1.5 py-0.5 rounded border font-bold uppercase tracking-widest ${badgeColor}`}>
            {badge}
          </span>
          <div className="flex flex-col">
            <span className="text-[10px] text-white/40 uppercase tracking-widest">{label}</span>
            <span className="text-[9px] text-white/10 font-mono italic">{sublabel}</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <select
          value={selectedFolder}
          onChange={(e) => onFolderChange(e.target.value)}
          className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-emerald-500/40"
        >
          <option value="">-- select study --</option>
          {folders.map((f) => (
            <option key={f.name} value={f.name}>{f.name} ({f.fileCount})</option>
          ))}
        </select>
        <select
          value={selectedFile}
          onChange={(e) => onFileChange(e.target.value)}
          disabled={!selectedFolder}
          className={`w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-emerald-500/40 ${!selectedFolder ? "opacity-30 cursor-not-allowed" : ""}`}
        >
          <option value="">-- select volume --</option>
          {files.map((f) => (<option key={f} value={f}>{f}</option>))}
        </select>
      </div>
      {resolvedPath && (
        <div className="flex items-center gap-2 group">
          <div className="text-[8px] font-bold text-white/10 uppercase tracking-widest whitespace-nowrap">Path Resolution</div>
          <div className="flex-1 h-px bg-white/[0.03]" />
          <div className="text-[9px] font-mono text-white/20 bg-black/20 px-3 py-1 rounded border border-white/[0.03] transition-all group-hover:border-white/10 group-hover:text-white/40">
            {resolvedPath}
          </div>
        </div>
      )}
    </div>
  );
}