"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

export default function RealignPage() {
    const [folders, setFolders] = useState<ConvertedFolder[]>([]);
    const [loadingFiles, setLoadingFiles] = useState(true);
    const [selectedFolder, setSelectedFolder] = useState("");
    const [selectedFile, setSelectedFile] = useState("");

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

    const handleFileChange = (file: string) => {
        setSelectedFile(file);
        setResult(null);
        setError(null);
        setStatus(null);
    };

    const resolvedFilename =
        selectedFolder && selectedFile ? `${selectedFolder}/${selectedFile}` : "";

    const handleRunRealign = async () => {
        if (!resolvedFilename) return;
        setIsLoading(true);
        setStatus("Realignment in progress...");
        setError(null);
        setResult(null);

        try {
            const response = await fetch("/api/realign", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: resolvedFilename }),
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || "Failed to run realignment");
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
                    <div className="inline-flex items-center gap-[7px] px-3 py-1 rounded-full border border-[#3b82f62e] bg-[#3b82f60d] mb-6">
                        <div className="badge-dot bg-blue-500 shadow-[0_0_8px_#3b82f6]" />
                        <span className="text-[#60a5fa] text-[12px] tracking-[0.15em] uppercase">
                            Step 03 · Realignment
                        </span>
                    </div>

                    <h1 className="text-[52px] font-black tracking-[-0.05em] leading-[0.95] mb-4">
                        Spatial <span className="text-blue-500">Realignment</span>
                    </h1>

                    <p className="text-white/35 text-[17px] leading-[1.6] font-light font-sans max-w-[500px]">
                        Corrects for intra-session head motion by estimating a 6-parameter rigid body 
                        transformation (3 translations, 3 rotations).
                    </p>
                </div>
            </section>

            {/* ────────────────────────────── CONTENT ─────────────────────────── */}
            <div className="relative max-w-[1400px] mx-auto px-6 py-10 space-y-10">
                
                {/* 01 - Data Source */}
                <section>
                    <div className="flex items-center gap-3 mb-6">
                        <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">03-A</span>
                        <div className="flex-1 h-px bg-white/[0.04]" />
                        <span className="text-white/25 text-[10px]">Data Source</span>
                    </div>

                    <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6">
                        {loadingFiles ? (
                            <div className="text-xs text-white/20 italic">Loading file inventory...</div>
                        ) : folders.length === 0 ? (
                            <div className="text-xs text-red-500/50">No studies found in converted repository.</div>
                        ) : (
                            <ImagePicker
                                badge="NIFTI"
                                badgeColor="text-blue-400 bg-blue-500/10 border-blue-500/20"
                                label="Target Series"
                                sublabel="Select the volume series for motion correction"
                                folders={folders}
                                selectedFolder={selectedFolder}
                                selectedFile={selectedFile}
                                onFolderChange={handleFolderChange}
                                onFileChange={handleFileChange}
                                resolvedPath={resolvedFilename ? `converted/${resolvedFilename}` : ""}
                            />
                        )}
                    </div>
                </section>

                {/* 02 - Configuration */}
                <section>
                    <div className="flex items-center gap-3 mb-6">
                        <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">03-B</span>
                        <div className="flex-1 h-px bg-white/[0.04]" />
                        <span className="text-white/25 text-[10px]">Realignment Settings</span>
                    </div>

                    <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6 grid grid-cols-2 gap-x-8 gap-y-6">
                        <StaticParam label="Estimation" value="Least Squares · Rigid Body" />
                        <StaticParam label="Reslicing" value="Interpolation · 4th Degree B-Spline" />
                        <StaticParam label="Flags" value="Register to First · Mean Image Created" />
                        <StaticParam label="Quality" value="SPM12 Default (0.9)" />
                    </div>
                </section>

                {/* Run Button */}
                <button
                    onClick={handleRunRealign}
                    disabled={isLoading || !selectedFolder || !selectedFile}
                    className={`
                        w-full py-4 rounded-xl border font-bold text-xs uppercase tracking-[0.2em] transition-all
                        ${isLoading || !selectedFolder || !selectedFile
                            ? "bg-white/5 border-white/5 text-white/10 cursor-not-allowed"
                            : "bg-blue-500/10 border-blue-500/10 text-blue-400 hover:bg-blue-500/15 hover:border-blue-500/20 shadow-[0_4px_20px_rgba(59,130,246,0.05)]"
                        }
                    `}
                >
                    {isLoading ? "Executing Step 03..." : "Initialize Realignment"}
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
                                    <span className="text-[10px] text-blue-400 font-bold uppercase tracking-widest">Realignment Complete</span>
                                    <span className="text-[9px] text-white/20 font-mono">Status: {status}</span>
                                </div>
                                <div className="text-[10px] font-mono bg-black/40 border border-white/[0.05] rounded-lg p-3 text-white/60">
                                    <span className="text-white/20 mr-2">Output:</span>
                                    {result.output}
                                </div>
                                {result.rp_file && (
                                    <div className="text-[9px] font-mono text-white/40">
                                        <span className="text-white/10 uppercase mr-2 tracking-widest">RP File:</span>
                                        {result.rp_file}
                                    </div>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

function StaticParam({ label, value }: { label: string; value: string }) {
    return (
        <div className="space-y-2">
            <label className="text-[10px] text-white/40 uppercase tracking-widest ml-1">{label}</label>
            <div className="w-full bg-black/20 border border-white/[0.03] rounded-lg px-3 py-2 text-[11px] font-mono text-white/30 italic">
                {value}
            </div>
        </div>
    );
}

interface ConvertedFolder {
    name: string;
    fileCount: number;
    files: string[];
}

// ── Image Picker helper (Premium Style) ──────────────────────────────
function ImagePicker({
  badge, badgeColor, label, sublabel,
  folders, selectedFolder, selectedFile,
  onFolderChange, onFileChange,
  resolvedPath
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
            <span className="text-[10px] text-white/40 uppercase tracking-widest">{label}</span>
            <span className="text-[9px] text-white/10 font-mono italic">{sublabel}</span>
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        <select
          value={selectedFolder}
          onChange={(e) => onFolderChange(e.target.value)}
          className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-blue-500/40"
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
          className={`w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-blue-500/40 ${!selectedFolder ? "opacity-30 cursor-not-allowed" : ""}`}
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
