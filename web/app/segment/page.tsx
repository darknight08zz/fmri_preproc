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

interface SegmentResult {
    output_files: {
        c1: string; c2: string; c3: string;
        m: string; y: string; iy: string;
    };
    time_load: number;
    time_bias: number;
    time_gmm: number;
    time_mrf: number;
    time_deform: number;
    time_save: number;
    time_total: number;
    n_gm_voxels: number;
    n_wm_voxels: number;
    n_csf_voxels: number;
    stdout?: string;
}

const TISSUE_LABELS = [
    { name: "Gray Matter", tpm: "TPM.nii,1", color: "text-[#a78bfa]", save: true },
    { name: "White Matter", tpm: "TPM.nii,2", color: "text-[#60a5fa]", save: true },
    { name: "CSF", tpm: "TPM.nii,3", color: "text-[#34d399]", save: true },
    { name: "Bone", tpm: "TPM.nii,4", color: "text-white/20", save: false },
    { name: "Soft Tissue", tpm: "TPM.nii,5", color: "text-white/20", save: false },
    { name: "Air/Background", tpm: "TPM.nii,6", color: "text-white/20", save: false },
];

export default function SegmentPage() {
    const [folders, setFolders] = useState<ConvertedFolder[]>([]);
    const [loadingFiles, setLoadingFiles] = useState(true);

    const [t1wFolder, setT1wFolder] = useState("");
    const [t1wFile, setT1wFile] = useState("");

    const [biasReg, setBiasReg] = useState("0.0001");
    const [biasFwhm, setBiasFwhm] = useState("60");
    const [nGauss, setNGauss] = useState([1, 1, 2, 3, 4, 2]);
    const [mrfStrength, setMrfStrength] = useState("1");
    const [samplingMm, setSamplingMm] = useState("3");

    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<string | null>(null);
    const [result, setResult] = useState<SegmentResult | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [showLog, setShowLog] = useState(false);

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

    const t1wPath = t1wFolder && t1wFile ? `converted/${t1wFolder}/${t1wFile}` : "";
    const canRun = !isLoading && !!t1wPath;

    const handleRun = async () => {
        if (!canRun) return;
        setIsLoading(true);
        setStatus("Segmenting brain tissues...");
        setError(null);
        setResult(null);
        setShowLog(false);

        try {
            const res = await fetch("/api/segment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    t1w_path: t1wPath,
                    bias_reg: parseFloat(biasReg),
                    bias_fwhm: parseFloat(biasFwhm),
                    mrf_strength: parseFloat(mrfStrength),
                    sampling_mm: parseFloat(samplingMm),
                    n_gauss: nGauss,
                }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error ?? "Segmentation failed");
            setResult(data as SegmentResult);
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
                    <div className="inline-flex items-center gap-[7px] px-3 py-1 rounded-full border border-[#a78bfa2e] bg-[#a78bfa0d] mb-6">
                        <div className="badge-dot bg-[#a78bfa] shadow-[0_0_8px_#a78bfa]" />
                        <span className="text-[#a78bfa] text-[12px] tracking-[0.15em] uppercase">
                            Step 04 · Segmentation
                        </span>
                    </div>

                    <h1 className="text-[52px] font-black tracking-[-0.05em] leading-[0.95] mb-4">
                        Tissue <span className="text-[#a78bfa]">Segmentation</span>
                    </h1>

                    <p className="text-white/35 text-[17px] leading-[1.6] font-light font-sans max-w-[500px]">
                        Partitions T1w scans into gray matter, white matter, and CSF while correcting for 
                        intensity non-uniformity and generating deformation fields.
                    </p>
                </div>
            </section>

            {/* ────────────────────────────── CONTENT ─────────────────────────── */}
            <div className="relative max-w-[1400px] mx-auto px-6 py-10 space-y-10">
                
                {/* 01 - Input Data */}
                <section>
                    <div className="flex items-center gap-3 mb-6">
                        <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">04-A</span>
                        <div className="flex-1 h-px bg-white/[0.04]" />
                        <span className="text-white/25 text-[10px]">Structural Input</span>
                    </div>

                    <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6 space-y-6">
                        <ImagePicker
                            badge="T1W"
                            badgeColor="text-[#a78bfa] bg-[#a78bfa1a] border-[#a78bfa33]"
                            label="Anatomical Volume"
                            sublabel="High-resolution T1-weighted image"
                            folders={folders}
                            selectedFolder={t1wFolder}
                            selectedFile={t1wFile}
                            onFolderChange={(f: string) => { setT1wFolder(f); setT1wFile(""); }}
                            onFileChange={setT1wFile}
                            resolvedPath={t1wPath}
                        />

                        <div className="flex items-center gap-3 px-4 py-3 bg-white/[0.02] border border-white/[0.05] rounded-lg">
                            <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                            <div className="flex flex-col">
                                <span className="text-[10px] text-white/40 uppercase tracking-widest">Tissue Probability Map</span>
                                <span className="text-[11px] font-mono text-white/60">SPM12 Bundled Assets (TPM.nii)</span>
                            </div>
                        </div>
                    </div>
                </section>

                {/* 02 - Channel Parameters */}
                <section>
                    <div className="flex items-center gap-3 mb-6">
                        <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">04-B</span>
                        <div className="flex-1 h-px bg-white/[0.04]" />
                        <span className="text-white/25 text-[10px]">Bias & Sampling</span>
                    </div>

                    <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl p-6 grid grid-cols-2 gap-x-8 gap-y-6">
                        <ParamInput label="Bias Regularisation" value={biasReg} onChange={setBiasReg} />
                        <div className="space-y-2">
                             <label className="text-[10px] text-white/40 uppercase tracking-widest ml-1">Bias FWHM</label>
                             <select
                                value={biasFwhm}
                                onChange={(e) => setBiasFwhm(e.target.value)}
                                className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#a78bfa4d]"
                             >
                                <option value="30">30 mm</option>
                                <option value="60">60 mm (SPM Default)</option>
                                <option value="90">90 mm</option>
                                <option value="Inf">No Correction</option>
                             </select>
                        </div>
                        <ParamInput label="MRF Strength" value={mrfStrength} onChange={setMrfStrength} />
                        <div className="space-y-2">
                             <label className="text-[10px] text-white/40 uppercase tracking-widest ml-1">Sampling Distance</label>
                             <select
                                value={samplingMm}
                                onChange={(e) => setSamplingMm(e.target.value)}
                                className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#a78bfa4d]"
                             >
                                <option value="1">1 mm (Accurate)</option>
                                <option value="2">2 mm</option>
                                <option value="3">3 mm (Fast/SPM)</option>
                             </select>
                        </div>
                    </div>
                </section>

                {/* 03 - Tissue Classes */}
                <section>
                    <div className="flex items-center gap-3 mb-6">
                        <span className="text-white/12 text-[10px] tracking-[0.18em] uppercase whitespace-nowrap">04-C</span>
                        <div className="flex-1 h-px bg-white/[0.04]" />
                        <span className="text-white/25 text-[10px]">Tissue Definition</span>
                    </div>

                    <div className="bg-white/[0.01] border border-white/[0.08] rounded-xl overflow-hidden">
                        <table className="w-full text-left">
                            <thead className="bg-white/[0.03] border-b border-white/[0.05]">
                                <tr>
                                    <th className="px-6 py-3 text-[9px] font-bold text-white/20 uppercase tracking-[0.2em]">Class</th>
                                    <th className="px-6 py-3 text-[9px] font-bold text-white/20 uppercase tracking-[0.2em] text-center">N Gauss</th>
                                    <th className="px-6 py-3 text-[9px] font-bold text-white/20 uppercase tracking-[0.2em] text-right">Output</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-white/[0.03]">
                                {TISSUE_LABELS.map((t, i) => (
                                    <tr key={i} className="hover:bg-white/[0.01] transition-colors">
                                        <td className="px-6 py-4">
                                            <div className="flex items-center gap-3">
                                                <div className={`w-1 h-3 rounded-full ${t.color.replace('text-', 'bg-')}`} />
                                                <span className={`text-[11px] font-mono ${t.color}`}>{t.name}</span>
                                            </div>
                                        </td>
                                        <td className="px-6 py-4 text-center">
                                            <input
                                                type="number"
                                                value={nGauss[i]}
                                                onChange={(e) => {
                                                    const next = [...nGauss];
                                                    next[i] = parseInt(e.target.value) || 1;
                                                    setNGauss(next);
                                                }}
                                                className="w-12 bg-black/40 border border-white/[0.05] rounded px-2 py-1 text-[10px] font-mono text-center focus:outline-none focus:border-[#a78bfa4d]"
                                            />
                                        </td>
                                        <td className="px-6 py-4 text-right">
                                            {t.save ? (
                                                <span className="text-[9px] text-emerald-400/60 font-mono tracking-widest">SAVE ACTIVE</span>
                                            ) : (
                                                <span className="text-[9px] text-white/10 font-mono tracking-widest">—</span>
                                            )}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </section>

                {/* Run Button */}
                <button
                    onClick={handleRun}
                    disabled={!canRun}
                    className={`
                        w-full py-4 rounded-xl border font-bold text-xs uppercase tracking-[0.2em] transition-all
                        ${!canRun
                            ? "bg-white/5 border-white/5 text-white/10 cursor-not-allowed"
                            : isLoading 
                                ? "bg-[#a78bfa1a] border-[#a78bfa33] text-[#a78bfa] cursor-wait"
                                : "bg-[#a78bfa1a] border-[#a78bfa33] text-[#a78bfa] hover:bg-[#a78bfa26] hover:border-[#a78bfa4d] shadow-[0_4px_20px_rgba(167,139,250,0.05)]"
                        }
                    `}
                >
                    {isLoading ? "Executing Segmentation... (15-25m)" : "Initialize Segmentation Pipeline"}
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
                                    <span className="text-[10px] text-[#a78bfa] font-bold uppercase tracking-widest">Segmentation Complete</span>
                                    <span className="text-[9px] text-white/20 font-mono">Status: {status}</span>
                                </div>
                                
                                <div className="grid grid-cols-3 gap-4">
                                    <StatCard label="GM Voxels" value={`${(result.n_gm_voxels/1000).toFixed(1)}k`} color="text-[#a78bfa]" />
                                    <StatCard label="WM Voxels" value={`${(result.n_wm_voxels/1000).toFixed(1)}k`} color="text-[#60a5fa]" />
                                    <StatCard label="CSF Voxels" value={`${(result.n_csf_voxels/1000).toFixed(1)}k`} color="text-emerald-400" />
                                </div>

                                <div className="space-y-1">
                                    <span className="text-[9px] text-white/20 uppercase tracking-[0.2em] ml-1">Generated Artifacts</span>
                                    {Object.entries(result.output_files).map(([key, path]) => (
                                        <div key={key} className="flex items-center gap-3 px-4 py-3 bg-black/40 border border-white/[0.05] rounded-lg">
                                            <span className="text-[10px] font-mono text-white/20 w-8">{key}</span>
                                            <span className="text-[11px] font-mono text-white/60 truncate flex-1">{String(path).split('/').pop()}</span>
                                            <span className="text-[9px] text-emerald-500 font-bold uppercase tracking-widest">READY</span>
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

function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
    return (
        <div className="bg-black/40 border border-white/[0.05] rounded-lg p-4 text-center">
            <div className={`text-xl font-black tracking-tight ${color}`}>{value}</div>
            <div className="text-[9px] text-white/20 uppercase tracking-[0.1em] mt-1">{label}</div>
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
                className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#a78bfa4d]"
            />
        </div>
    );
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
          className="w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#a78bfa4d]"
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
          className={`w-full bg-black/40 border border-white/[0.05] rounded-lg px-3 py-2 text-[11px] font-mono text-white/60 focus:outline-none focus:border-[#a78bfa4d] ${!selectedFolder ? "opacity-30 cursor-not-allowed" : ""}`}
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