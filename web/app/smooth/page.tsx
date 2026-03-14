"use client";

import { useState, useEffect } from "react";
import { Button } from "../../components/ui/button";
import { Label } from "../../components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "../../components/ui/card";
import { Badge } from "../../components/ui/badge";
import { Separator } from "../../components/ui/separator";
import NiftiViewer from "../../components/NiftiViewer";

interface SmoothResult {
    outputFile: string;
    timeTotal: number;
    log: string;
}

interface ConvertedFolder {
    name: string;
    fileCount: number;
    files: string[];
}

const FIELD =
    "w-full px-3 py-2 bg-[#1a1b22] text-white border border-[#2d2f3d] rounded-md " +
    "focus:outline-none focus:border-[#2563eb] font-mono text-sm transition-colors";

const SELECT = FIELD;

export default function SmoothPage() {
    const [folders, setFolders] = useState<ConvertedFolder[]>([]);
    const [loadingFiles, setLoadingFiles] = useState(true);

    // Image selection state
    const [funcFolder, setFuncFolder] = useState("");
    const [funcFile, setFuncFile] = useState("");

    const [fwhm, setFwhm] = useState<number>(6);
    const [running, setRunning] = useState(false);
    const [result, setResult] = useState<SmoothResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [log, setLog] = useState("");

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

    const resolvePath = (folder: string, file: string) =>
        folder && file ? `converted/${folder}/${file}` : "";

    const funcPath = resolvePath(funcFolder, funcFile);

    async function handleRun() {
        if (!funcPath) return;
        setRunning(true);
        setError(null);
        setResult(null);
        setLog("");

        try {
            const res = await fetch("/api/smooth", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ funcPath, fwhm: [fwhm, fwhm, fwhm] }),
            });
            const data = await res.json();

            if (!res.ok || !data.success) {
                setError(data.error || data.details || "Smoothing failed");
                setLog(data.details || "");
            } else {
                setResult({ outputFile: data.outputFile, timeTotal: data.timeTotal, log: data.log });
                setLog(data.log);
            }
        } catch (e) {
            setError(String(e));
        } finally {
            setRunning(false);
        }
    }

  return (
    <div className="min-h-[90vh] text-white">
      {/* ────────────────────────────── HERO ───────────────────────────── */}
      <section className="relative overflow-hidden border-b border-white/[0.04] py-10 px-6">
        <div className="grid-bg" />
        <div className="glow-orb" />
        
        <div className="relative max-w-[1400px] mx-auto">
          {/* Badge */}
          <div className="inline-flex items-center gap-[7px] px-3 py-1 rounded-full border border-[#10b9812e] bg-[#10b9810d] mb-6">
            <div className="badge-dot" />
            <span className="text-[#34d399] text-[12px] tracking-[0.15em] uppercase">
              Step 06 · Gaussian Smoothing
            </span>
          </div>

          <h1 className="text-[52px] font-black tracking-[-0.05em] leading-[0.95] mb-4">
            Spatial <span className="text-[#34d399]">Smoothing</span>
          </h1>

          <p className="text-white/35 text-[17px] leading-[1.6] font-light font-sans max-w-[500px]">
             Suppresses high-frequency noise and increases signal-to-noise ratio by applying 
             a 3D Gaussian kernel. Essential for inter-subject alignment and statistical validity.
          </p>
        </div>
      </section>

      {/* ────────────────────────────── CONTENT ─────────────────────────── */}
      <div className="relative max-w-[1400px] mx-auto px-6 py-10 space-y-10">
        
        {/* Settings */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">06-A</span>
            <div className="flex-1 h-px bg-white/[0.04]" />
            <span className="text-white/25 text-[10px]">Smoothing Parameters</span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "FWHM Kernel", val: `${fwhm} × ${fwhm} × ${fwhm} mm` },
              { label: "Data Type", val: "Float32" },
              { label: "Prefix", val: "s*" },
              { label: "Algorithm", val: "3D Gaussian" },
            ].map((s) => (
              <div key={s.label} className="bg-white/[0.01] border border-white/[0.05] rounded-xl p-4 transition-all hover:bg-white/[0.02]">
                <div className="text-white font-bold text-sm mb-1">{s.val}</div>
                <div className="text-white/15 text-[8px] uppercase tracking-widest">{s.label}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Inputs */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">06-B</span>
            <div className="flex-1 h-px bg-white/[0.04]" />
            <span className="text-white/25 text-[10px]">Input Configuration</span>
          </div>

          <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6 space-y-8">
            <div className="space-y-6">
               {/* FWHM Slider */}
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex flex-col">
                    <span className="text-[10px] text-white/40 uppercase tracking-widest">Gaussian Kernel Size</span>
                    <span className="text-[9px] text-white/10 font-mono italic">Full Width at Half Maximum (FWHM) in mm</span>
                  </div>
                  <span className="text-lg font-black text-emerald-400 font-mono">{fwhm} mm</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="12"
                  step="1"
                  value={fwhm}
                  onChange={(e) => setFwhm(Number(e.target.value))}
                  className="w-full h-1 bg-white/5 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                />
                <div className="flex justify-between text-[8px] text-white/10 font-mono uppercase">
                  <span>Narrow (2mm)</span>
                  <span>Standard (6-8mm)</span>
                  <span>Wide (12mm)</span>
                </div>
              </div>

              <div className="h-px bg-white/[0.03]" />

              {/* Image Picker */}
              <ImagePicker
                badge="INPUT"
                badgeColor="text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
                label="Normalized Volume"
                sublabel="wrarfunc_4D.nii — Pre-smoothed volume"
                folders={folders}
                selectedFolder={funcFolder}
                selectedFile={funcFile}
                onFolderChange={(f) => { setFuncFolder(f); setFuncFile(""); }}
                onFileChange={setFuncFile}
                resolvedPath={funcPath}
              />
            </div>

            <button
              onClick={handleRun}
              disabled={running || !funcFolder || !funcFile}
              className={`
                w-full py-4 rounded-xl border font-bold text-xs uppercase tracking-[0.2em] transition-all
                ${running || !funcFolder || !funcFile
                  ? "bg-white/5 border-white/5 text-white/10 cursor-not-allowed"
                  : "bg-emerald-500/10 border-emerald-500/10 text-emerald-400 hover:bg-emerald-500/15 hover:border-emerald-500/20 shadow-[0_4px_20px_rgba(16,185,129,0.05)]"
                }
              `}
            >
              {running ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Applying Gaussian Kernel...
                </span>
              ) : "Initialize Pipeline Step 06"}
            </button>
          </div>
        </section>

        {/* Results Area */}
        {(error || result) && (
          <section className="animate-in fade-in slide-in-from-bottom-2 duration-500">
             <div className="flex items-center gap-3 mb-6">
              <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">06-C</span>
              <div className="flex-1 h-px bg-white/[0.04]" />
              <span className={`${error ? 'text-red-400' : 'text-emerald-400'} text-[10px]`}>
                {error ? 'Process Error' : 'Smoothing Complete'}
              </span>
            </div>

            {error ? (
              <div className="bg-red-500/5 border border-red-500/20 rounded-xl p-5">
                <div className="text-[10px] text-red-400 font-mono whitespace-pre-wrap">{error}</div>
              </div>
            ) : result && (
               <div className="space-y-6">
                <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
                  <div>
                    <div className="text-xs font-mono text-emerald-400 font-bold mb-1">SIGNAL: ANALYSIS READY</div>
                    <div className="text-[10px] text-white/40 font-mono">FINALISED VOLUME: {result.outputFile?.split(/[\\/]/).pop()}</div>
                  </div>
                  <div className="text-[10px] font-mono font-bold text-white/60 bg-white/5 py-1 px-3 rounded-full border border-white/5">
                    EXECUTION TIME: {result.timeTotal?.toFixed(2)}S
                  </div>
                </div>

                {/* Viewer */}
                <div className="border border-white/[0.08] rounded-xl overflow-hidden bg-black/40">
                  <div className="px-4 py-3 bg-white/[0.02] border-b border-white/[0.08] flex items-center justify-between">
                    <span className="text-[9px] font-bold text-white/30 uppercase tracking-widest">Smoothed Output Preview</span>
                    <div className="flex gap-1">
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 opacity-40" />
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 opacity-20" />
                    </div>
                  </div>
                  <NiftiViewer
                    key={result.outputFile}
                    url={`/api/serve-file?path=${result.outputFile}`}
                  />
                </div>
              </div>
            )}
          </section>
        )}

        {/* Log */}
        {log && (
          <section>
             <div className="flex items-center gap-3 mb-4">
              <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">Process Log</span>
              <div className="flex-1 h-px bg-white/[0.04]" />
            </div>
            <div className="bg-[#0d0e11] border border-white/[0.05] rounded-xl p-5 overflow-hidden">
               <pre className="text-[9px] text-white/30 font-mono whitespace-pre-wrap max-h-64 overflow-y-auto no-scrollbar">
                {log}
              </pre>
            </div>
          </section>
        )}

        {/* Bottom Decoration */}
        <div className="flex items-center justify-center gap-2 py-10">
          <div className="h-px bg-emerald-500/20 w-12" />
          <span className="text-white/5 text-[9px] tracking-[0.3em] uppercase">Pipeline Finalised</span>
          <div className="h-px bg-emerald-500/20 w-12" />
        </div>
      </div>
    </div>
  );
}

// ── Image Picker helper (Premium Style) ──────────────────────────────
interface ImagePickerProps {
  badge: string;
  badgeColor: string;
  label: string;
  sublabel: string;
  folders: ConvertedFolder[];
  selectedFolder: string;
  selectedFile: string;
  onFolderChange: (f: string) => void;
  onFileChange: (f: string) => void;
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
        <div className="space-y-1.5">
          <label className="text-[8px] text-white/15 uppercase tracking-wider ml-1">Study Folder</label>
          <select
            value={selectedFolder}
            onChange={(e) => onFolderChange(e.target.value)}
            className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-emerald-500/40"
          >
            <option value="">-- select study --</option>
            {folders.map((f) => (
              <option key={f.name} value={f.name}>
                {f.name} ({f.fileCount})
              </option>
            ))}
          </select>
        </div>

        <div className="space-y-1.5">
          <label className="text-[8px] text-white/15 uppercase tracking-wider ml-1">NIfTI Volume</label>
          <select
            value={selectedFile}
            onChange={(e) => onFileChange(e.target.value)}
            disabled={!selectedFolder}
            className={`
              w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-emerald-500/40
              ${!selectedFolder ? "opacity-30 cursor-not-allowed" : ""}
            `}
          >
            <option value="">-- select volume --</option>
            {files.map((f) => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </div>
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
