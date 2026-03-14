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
}

interface CoregResult {
  params:        number[];          // [tx, ty, tz, rx, ry, rz]
  M:             number[][];        // 4×4 matrix
  output_files:  string[];
  time_estimate: number;
  time_reslice:  number;
  stdout?:       string;
}

export default function CoregPage() {

  // ── folder / file lists ───────────────────
  const [folders,      setFolders]      = useState<ConvertedFolder[]>([]);
  const [loadingFiles, setLoadingFiles] = useState(true);

  // ── image selection ───────────────────────
  const [refFolder,   setRefFolder]   = useState("");
  const [refFile,     setRefFile]     = useState("");
  const [srcFolder,   setSrcFolder]   = useState("");
  const [srcFile,     setSrcFile]     = useState("");
  const [otherFolder, setOtherFolder] = useState("");
  const [otherFile,   setOtherFile]   = useState("");

  const [sep0,  setSep0]  = useState("4");
  const [sep1,  setSep1]  = useState("2");
  const [fwhm0, setFwhm0] = useState("7");
  const [fwhm1, setFwhm1] = useState("7");

  const [interp, setInterp] = useState("4");
  const [mask,   setMask]   = useState(false);
  const [prefix, setPrefix] = useState("r");

  const [isLoading, setIsLoading] = useState(false);
  const [status,    setStatus]    = useState<string | null>(null);
  const [result,    setResult]    = useState<CoregResult | null>(null);
  const [error,     setError]     = useState<string | null>(null);
  const [showLog,   setShowLog]   = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const res  = await fetch("/api/converted-files");
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

  const refPath   = resolvePath(refFolder,   refFile);
  const srcPath   = resolvePath(srcFolder,   srcFile);
  const otherPath = resolvePath(otherFolder, otherFile);

  const canRun = !isLoading && !!refPath && !!srcPath;

  const handleRun = async () => {
    if (!canRun) return;

    setIsLoading(true);
    setStatus("Aligning volumes...");
    setError(null);
    setResult(null);
    setShowLog(false);

    try {
      const body: Record<string, any> = {
        ref_path:        refPath,
        source_path:     srcPath,
        other_paths:     otherPath ? [otherPath] : [],
        separation:      [parseFloat(sep0), parseFloat(sep1)],
        hist_smooth_fwhm:[parseFloat(fwhm0), parseFloat(fwhm1)],
        interp_order:    parseInt(interp),
        wrap:            [0, 0, 0],
        mask,
        prefix,
      };

      const res  = await fetch("/api/coreg", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(body),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Coregistration failed");

      setResult(data as CoregResult);
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
          <div className="inline-flex items-center gap-[7px] px-3 py-1 rounded-full border border-[#818cf82e] bg-[#818cf80d] mb-6">
            <div className="badge-dot bg-[#818cf8] shadow-[0_0_8px_#818cf8]" />
            <span className="text-[#818cf8] text-[12px] tracking-[0.15em] uppercase">
                Step 05 · Coregistration
            </span>
          </div>

          <h1 className="text-[52px] font-black tracking-[-0.05em] leading-[0.95] mb-4">
            Volume <span className="text-[#818cf8]">Coregistration</span>
          </h1>

          <p className="text-white/35 text-[17px] leading-[1.6] font-light font-sans max-w-[500px]">
            Aligns images of different modalities (e.g., Mean Functional to T1-Structural) 
            using Normalized Mutual Information.
          </p>
        </div>
      </section>

      {/* ────────────────────────────── CONTENT ─────────────────────────── */}
      <div className="relative max-w-[1400px] mx-auto px-6 py-10 space-y-10">
        
        {/* 01 - Selection */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">05-A</span>
            <div className="flex-1 h-px bg-white/[0.04]" />
            <span className="text-white/25 text-[10px]">Image Selection</span>
          </div>

          <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6 space-y-8">
            <ImagePicker
                badge="REF" badgeColor="text-[#818cf8] bg-[#818cf81a] border-[#818cf833]"
                label="Reference Image" sublabel="T1w Structural (Fixed Target)"
                folders={folders} selectedFolder={refFolder} selectedFile={refFile}
                onFolderChange={(f: string) => { setRefFolder(f); setRefFile(""); }} onFileChange={setRefFile}
                resolvedPath={refPath}
            />
            <div className="h-px bg-white/[0.04] sm:mx-10" />
            <ImagePicker
                badge="SRC" badgeColor="text-emerald-400 bg-emerald-500/10 border-emerald-500/20"
                label="Source Image" sublabel="Mean Functional (Moving Object)"
                folders={folders} selectedFolder={srcFolder} selectedFile={srcFile}
                onFolderChange={(f: string) => { setSrcFolder(f); setSrcFile(""); }} onFileChange={setSrcFile}
                resolvedPath={srcPath}
            />
            <div className="h-px bg-white/[0.04] sm:mx-10" />
            <ImagePicker
                badge="OTH" badgeColor="text-amber-400 bg-amber-500/10 border-amber-500/20"
                label="Other Images" sublabel="Realigned 4D series (Receives transform)"
                folders={folders} selectedFolder={otherFolder} selectedFile={otherFile}
                onFolderChange={(f: string) => { setOtherFolder(f); setOtherFile(""); }} onFileChange={setOtherFile}
                resolvedPath={otherPath} optional
            />
          </div>
        </section>

        {/* 02 - Estimation */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">05-B</span>
            <div className="flex-1 h-px bg-white/[0.04]" />
            <span className="text-white/25 text-[10px]">Estimation Strategy</span>
          </div>

          <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6 grid grid-cols-2 gap-x-8 gap-y-6">
            <div className="col-span-2 space-y-2">
                <label className="text-[10px] text-white/40 uppercase tracking-widest ml-1">Separation (Coarse → Fine)</label>
                <div className="grid grid-cols-2 gap-4">
                    <input value={sep0} onChange={e => setSep0(e.target.value)} className="bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#818cf84d]" />
                    <input value={sep1} onChange={e => setSep1(e.target.value)} className="bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#818cf84d]" />
                </div>
            </div>
            <div className="col-span-2 space-y-2">
                <label className="text-[10px] text-white/40 uppercase tracking-widest ml-1">Histogram Smoothing FWHM</label>
                <div className="grid grid-cols-2 gap-4">
                    <input value={fwhm0} onChange={e => setFwhm0(e.target.value)} className="bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#818cf84d]" />
                    <input value={fwhm1} onChange={e => setFwhm1(e.target.value)} className="bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#818cf84d]" />
                </div>
            </div>
          </div>
        </section>

        {/* 03 - Reslice */}
        <section>
          <div className="flex items-center gap-3 mb-6">
            <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">05-C</span>
            <div className="flex-1 h-px bg-white/[0.04]" />
            <span className="text-white/25 text-[10px]">Reslicing Configuration</span>
          </div>

          <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6 grid grid-cols-2 gap-x-8 gap-y-6">
            <div className="space-y-2">
                <label className="text-[10px] text-white/40 uppercase tracking-widest ml-1">Interpolation Order</label>
                <select value={interp} onChange={e => setInterp(e.target.value)} className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#818cf84d]">
                    <option value="1">Trilinear (1st Degree)</option>
                    <option value="4">B-Spline (4th Degree · SPM Default)</option>
                    <option value="7">B-Spline (7th Degree · Max Precision)</option>
                </select>
            </div>
            <ParamInput label="Output Prefix" value={prefix} onChange={setPrefix} />
          </div>
        </section>

        <button
            onClick={handleRun}
            disabled={!canRun || isLoading}
            className={`
                w-full py-4 rounded-xl border font-bold text-xs uppercase tracking-[0.2em] transition-all
                ${!canRun || isLoading
                    ? "bg-white/5 border-white/5 text-white/10 cursor-not-allowed"
                    : "bg-[#818cf81a] border-[#818cf833] text-[#818cf8] hover:bg-[#818cf826] hover:border-[#818cf84d] shadow-[0_4px_20px_rgba(129,140,248,0.05)]"
                }
            `}
        >
            {isLoading ? "Executing Coregistration..." : "Initialize Coreg Pipeline"}
        </button>

        {/* Results/Error */}
        {(error || result) && (
            <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6">
                {error && (
                    <div className="text-[10px] text-red-400 font-mono whitespace-pre-wrap">{error}</div>
                )}
                {result && (
                    <div className="space-y-6">
                        <div className="flex items-center justify-between">
                            <span className="text-[10px] text-[#818cf8] font-bold uppercase tracking-widest">Alignment Complete</span>
                            <span className="text-[9px] text-white/20 font-mono">Total Execution: {(result.time_estimate + result.time_reslice).toFixed(1)}s</span>
                        </div>
                        
                        <div className="grid grid-cols-2 gap-4">
                            <div className="bg-black/40 border border-white/[0.05] rounded-lg p-4">
                                <span className="text-[9px] text-white/20 uppercase tracking-[0.1em] block mb-3">Rigid-Body Parameters</span>
                                <div className="grid grid-cols-3 gap-2 text-[10px] font-mono">
                                    {result.params.map((p, i) => (
                                        <div key={i} className="flex flex-col">
                                            <span className="text-[8px] text-white/10 uppercase font-sans tracking-widest mb-0.5">
                                                {['tx','ty','tz','rx','ry','rz'][i]}
                                            </span>
                                            <span className={i < 3 ? "text-blue-400" : "text-[#a78bfa]"}>
                                                {i < 3 ? p.toFixed(3) : (p * 180 / Math.PI).toFixed(2) + '°'}
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div className="bg-black/40 border border-white/[0.05] rounded-lg p-4">
                                <span className="text-[9px] text-white/20 uppercase tracking-[0.1em] block mb-3">Affine Transform (M)</span>
                                <div className="grid grid-cols-4 gap-1 text-[8px] font-mono text-white/40 leading-tight">
                                    {result.M.flat().map((v, i) => (
                                        <span key={i} className={Math.floor(i/4) === (i%4) ? "text-[#818cf8]" : ""}>{v.toFixed(3)}</span>
                                    ))}
                                </div>
                            </div>
                        </div>

                        <div className="space-y-1">
                            <span className="text-[9px] text-white/20 uppercase tracking-[0.2em] ml-1">Output Series</span>
                            {result.output_files.map((path, i) => (
                                <div key={i} className="flex items-center gap-3 px-4 py-3 bg-black/40 border border-white/[0.05] rounded-lg">
                                    <span className="text-[10px] font-mono text-white/20 w-8">r{i+1}</span>
                                    <span className="text-[11px] font-mono text-white/60 truncate flex-1">{path.split('/').pop()}</span>
                                    <span className="text-[9px] text-emerald-500 font-bold uppercase tracking-widest">SAVED</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        )}

      </div>
    </div>
  );
}

function ParamInput({ label, value, onChange }: any) {
    return (
        <div className="space-y-2">
            <label className="text-[10px] text-white/40 uppercase tracking-widest ml-1">{label}</label>
            <input
                value={value}
                onChange={(e) => onChange(e.target.value)}
                className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#818cf84d]"
            />
        </div>
    );
}

function ImagePicker({
  badge, badgeColor, label, sublabel,
  folders, selectedFolder, selectedFile,
  onFolderChange, onFileChange,
  resolvedPath, optional = false
}: any) {
  const files = folders.find((f: any) => f.name === selectedFolder)?.files ?? [];
  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <span className={`text-[9px] px-1.5 py-0.5 rounded border font-bold uppercase tracking-widest ${badgeColor}`}>
            {badge}
          </span>
          <div className="flex flex-col">
            <span className="text-[10px] text-white/40 uppercase tracking-widest">
                {label} {optional && <span className="text-white/10 font-normal ml-2">Optional</span>}
            </span>
            <span className="text-[9px] text-white/10 font-mono italic">{sublabel}</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <select
          value={selectedFolder}
          onChange={(e) => onFolderChange(e.target.value)}
          className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-white/10"
        >
          <option value="">-- select study --</option>
          {folders.map((f: any) => (
            <option key={f.name} value={f.name}>{f.name} ({f.fileCount})</option>
          ))}
        </select>
        <select
          value={selectedFile}
          onChange={(e) => onFileChange(e.target.value)}
          disabled={!selectedFolder}
          className={`w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-white/10 ${!selectedFolder ? "opacity-30 cursor-not-allowed" : ""}`}
        >
          <option value="">-- select volume --</option>
          {files.map((f: string) => (<option key={f} value={f}>{f}</option>))}
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