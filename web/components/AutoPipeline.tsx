"use client";

import { useState, useEffect, useCallback } from "react";
import { Loader2, CheckCircle2, Circle, XCircle, Play, Eye } from "lucide-react";
import Link from "next/link";

interface StepStatus {
    id: string;
    label: string;
    description: string;
    status: "pending" | "running" | "success" | "error";
    error?: string;
    output?: string;
}

interface AutoPipelineProps {
    folderName: string; // The baseline folder name (e.g., "Patient_01")
    onComplete?: () => void;
}

export default function AutoPipeline({ folderName, onComplete }: AutoPipelineProps) {
    const [steps, setSteps] = useState<StepStatus[]>([
        { id: "convert_anat", label: "Convert Anatomy", description: "DICOM to NIfTI (Structural)", status: "pending" },
        { id: "convert_func", label: "Convert Function", description: "DICOM to NIfTI (4D Functional)", status: "pending" },
        { id: "realign",      label: "Realign",          description: "Motion Correction", status: "pending" },
        { id: "stc",          label: "Slice Timing",    description: "Temporal Correction", status: "pending" },
        { id: "coreg",        label: "Coregistration",   description: "Func → Structural Alignment", status: "pending" },
        { id: "segment",      label: "Segmentation",     description: "Tissue Mapping & Warping", status: "pending" },
        { id: "normalise",    label: "Normalisation",    description: "MNI Space Normalization", status: "pending" },
        { id: "smooth",       label: "Smoothing",        description: "Gaussian Kernel (6mm)", status: "pending" },
    ]);

    const [isProcessing, setIsProcessing] = useState(false);
    const [currentStepIdx, setCurrentStepIdx] = useState(-1);
    const [logs, setLogs] = useState<string[]>([]);
    const [overallStatus, setOverallStatus] = useState<"idle" | "running" | "success" | "error">("idle");

    const addLog = (msg: string) => {
        setLogs(prev => [...prev.slice(-4), `[${new Date().toLocaleTimeString()}] ${msg}`]);
    };

    const updateStep = (id: string, updates: Partial<StepStatus>) => {
        setSteps(prev => prev.map(s => s.id === id ? { ...s, ...updates } : s));
    };

    const runPipeline = async () => {
        if (isProcessing) return;
        setIsProcessing(true);
        setOverallStatus("running");
        setLogs([]);
        
        // Reset steps
        setSteps(prev => prev.map(s => ({ ...s, status: "pending", error: undefined, output: undefined })));

        try {
            // STEP 1: Convert Anat
            setCurrentStepIdx(0);
            updateStep("convert_anat", { status: "running" });
            addLog("Starting Anatomy conversion...");
            const resAnat = await fetch("/api/convert", {
                method: "POST",
                body: JSON.stringify({ folderName, type: "anat" }),
            });
            if (!resAnat.ok) throw new Error("Anatomy conversion failed");
            updateStep("convert_anat", { status: "success" });

            // STEP 2: Convert Func
            setCurrentStepIdx(1);
            updateStep("convert_func", { status: "running" });
            addLog("Starting Functional conversion...");
            const resFunc = await fetch("/api/convert", {
                method: "POST",
                body: JSON.stringify({ folderName, type: "func" }),
            });
            if (!resFunc.ok) throw new Error("Functional conversion failed");
            updateStep("convert_func", { status: "success" });

            // Fetch file names for subsequent steps
            addLog("Fetching filenames...");
            const resFiles = await fetch("/api/converted-files");
            const dataFiles = await resFiles.json();
            const folderData = dataFiles.folders?.find((f: any) => f.name === folderName);
            if (!folderData) throw new Error("Could not find converted folder data");

            const anatFile = folderData.files.find((f: string) => f.startsWith("anat/") && f.endsWith(".nii"));
            const funcFile = folderData.files.find((f: string) => f.startsWith("func/") && f.endsWith(".nii"));

            if (!anatFile || !funcFile) throw new Error("Anat or Func NIfTI not found");

            // STEP 3: Realign
            setCurrentStepIdx(2);
            updateStep("realign", { status: "running" });
            addLog("Running Realign...");
            const resRealign = await fetch("/api/realign", {
                method: "POST",
                body: JSON.stringify({ filename: `${folderName}/${funcFile}` }),
            });
            const dataRealign = await resRealign.json();
            if (!resRealign.ok) throw new Error(dataRealign.error || "Realign failed");
            updateStep("realign", { status: "success", output: dataRealign.outputFile });
            const realignedFile = dataRealign.outputFile; // usually 'Patient_XX/func/r...'
            if (!realignedFile) throw new Error("Realign output file not found in response");

            // STEP 4: Slice Timing
            setCurrentStepIdx(3);
            updateStep("stc", { status: "running" });
            addLog("Running Slice Timing Correction...");
            
            // First extract params
            const jsonFile = funcFile.replace(".nii", ".json");
            const resParams = await fetch("/api/extract-stc-params", {
                method: "POST",
                body: JSON.stringify({ json_path: `${folderName}/${jsonFile}` }),
            });
            const stcParams = await resParams.json();
            if (!resParams.ok) throw new Error("Could not extract STC parameters");

            const resStc = await fetch("/api/stc", {
                method: "POST",
                body: JSON.stringify({
                    filename: realignedFile, // Use as-is from realign output
                    tr: stcParams.TR,
                    slices: stcParams.nSlices,
                    ta: stcParams.TA,
                    slice_order: Array.isArray(stcParams.sliceOrder) ? stcParams.sliceOrder.join(",") : stcParams.sliceOrder,
                    ref_slice: stcParams.refSlice
                }),
            });
            const dataStc = await resStc.json();
            if (!resStc.ok) throw new Error(dataStc.error || "STC failed");
            updateStep("stc", { status: "success", output: dataStc.outputFile });
            const stcFile = dataStc.outputFile; // usually 'func/sr...'

            // STEP 5: Coregister
            setCurrentStepIdx(4);
            updateStep("coreg", { status: "running" });
            addLog("Running Coregistration...");
            const resCoreg = await fetch("/api/coreg", {
                method: "POST",
                body: JSON.stringify({ ref_path: `converted/${folderName}/${anatFile}`, source_path: `converted/${stcFile}` }),
            });
            const dataCoreg = await resCoreg.json();
            if (!resCoreg.ok) throw new Error(dataCoreg.error || "Coregistration failed");
            updateStep("coreg", { status: "success", output: dataCoreg.outputFile });
            // Coreg reslices the functional image to match the anatomical
            const coregFuncFile = dataCoreg.outputFile; // usually 'func/r...'
            if (!coregFuncFile) throw new Error("Coregistration output file not found");

            // STEP 6: Segment
            setCurrentStepIdx(5);
            updateStep("segment", { status: "running" });
            addLog("Running Segmentation...");
            const resSeg = await fetch("/api/segment", {
                method: "POST",
                // CRITICAL: Segmentation MUST use the anatomical image (3D), not the functional (4D)
                body: JSON.stringify({ t1w_path: `converted/${folderName}/${anatFile}` }),
            });
            const dataSeg = await resSeg.json();
            if (!resSeg.ok) throw new Error(dataSeg.error || "Segmentation failed");
            updateStep("segment", { status: "success" });
            
            // Deformation matches the structural passed to segment (y_ + basename)
            const defField = dataSeg.output_files?.y || `${folderName}/anat/y_${pathBasename(anatFile)}`;

            // STEP 7: Normalise
            setCurrentStepIdx(6);
            updateStep("normalise", { status: "running" });
            addLog("Running Normalisation...");
            const resNorm = await fetch("/api/normalise", {
                method: "POST",
                // CRITICAL: Normalization MUST use the coregistered (resliced) functional, not the raw STC one
                body: JSON.stringify({ yPath: `converted/${defField}`, funcPath: `converted/${coregFuncFile}` }),
            });
            const dataNorm = await resNorm.json();
            if (!resNorm.ok) throw new Error(dataNorm.error || "Normalisation failed");
            updateStep("normalise", { status: "success", output: dataNorm.outputFile });
            const normFile = dataNorm.outputFile; // usually 'func/wsr...'

            // STEP 8: Smoothing
            setCurrentStepIdx(7);
            updateStep("smooth", { status: "running" });
            addLog("Running Smoothing...");
            const resSmooth = await fetch("/api/smooth", {
                method: "POST",
                body: JSON.stringify({ funcPath: `converted/${normFile}`, fwhm: [6, 6, 6] }),
            });
            const dataSmooth = await resSmooth.json();
            if (!resSmooth.ok) throw new Error(dataSmooth.error || "Smoothing failed");
            updateStep("smooth", { status: "success", output: dataSmooth.outputFile });

            addLog("Pipeline completed successfully!");
            setOverallStatus("success");
            if (onComplete) onComplete();

        } catch (err: any) {
            console.error(err);
            addLog(`ERROR: ${err.message}`);
            setOverallStatus("error");
            // Highlight current step as error
            setSteps(prev => {
                const updated = [...prev];
                if (currentStepIdx >= 0) {
                    updated[currentStepIdx] = { ...updated[currentStepIdx], status: "error", error: err.message };
                }
                return updated;
            });
        } finally {
            setIsProcessing(false);
        }
    };

    const pathBasename = (path: string) => path.split("/").pop() || path;

    return (
        <div className="bg-[#0f1117] border border-white/[0.05] rounded-2xl overflow-hidden shadow-2xl">
            {/* Header */}
            <div className="bg-white/[0.02] border-b border-white/[0.05] p-5 flex items-center justify-between">
                <div>
                    <h3 className="text-white font-bold text-sm tracking-tight flex items-center gap-2">
                        <Play className="w-3 h-3 text-emerald-400 fill-emerald-400" />
                        One-Click Pipeline
                    </h3>
                    <p className="text-white/30 text-[10px] uppercase tracking-widest mt-1">
                        Folder: <span className="text-white/60">{folderName}</span>
                    </p>
                </div>

                <button
                    onClick={runPipeline}
                    disabled={isProcessing || !folderName}
                    className={`
                        px-4 py-2 rounded-lg text-[11px] font-bold uppercase tracking-widest transition-all
                        ${isProcessing || !folderName
                            ? "bg-white/5 text-white/20 cursor-not-allowed"
                            : "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 hover:bg-emerald-500/20 hover:scale-[1.02]"
                        }
                    `}
                >
                    {isProcessing ? "Processing..." : "Start Pipeline"}
                </button>
            </div>

            <div className="p-6 grid md:grid-cols-2 gap-8">
                {/* Steps List */}
                <div className="space-y-4">
                    {steps.map((step, idx) => (
                        <div key={step.id} className="flex gap-4 group">
                            <div className="flex flex-col items-center">
                                <div className={`
                                    w-6 h-6 rounded-full flex items-center justify-center transition-all
                                    ${step.status === "success" ? "bg-emerald-500/20 text-emerald-500 border border-emerald-500/20" :
                                      step.status === "running" ? "bg-blue-500/20 text-blue-400 border border-blue-400/20 animate-pulse" :
                                      step.status === "error"   ? "bg-red-500/20 text-red-500 border border-red-500/20" :
                                      "bg-white/5 text-white/20 border border-white/5"}
                                `}>
                                    {step.status === "success" ? <CheckCircle2 className="w-3.5 h-3.5" /> :
                                     step.status === "running" ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> :
                                     step.status === "error"   ? <XCircle className="w-3.5 h-3.5" /> :
                                     <Circle className="w-3.5 h-3.5" />}
                                </div>
                                {idx < steps.length - 1 && (
                                    <div className="w-[1px] h-full bg-white/[0.05] mt-1 mb-1" />
                                )}
                            </div>
                            <div className="pb-4">
                                <div className={`text-[12px] font-bold tracking-tight transition-colors ${
                                    step.status === "running" ? "text-blue-400" :
                                    step.status === "success" ? "text-white/80" :
                                    step.status === "error"   ? "text-red-400" : "text-white/20"
                                }`}>
                                    {step.label}
                                </div>
                                <div className="text-[10px] text-white/10 group-hover:text-white/20 transition-colors uppercase tracking-wider">
                                    {step.description}
                                </div>
                                {step.error && (
                                    <div className="mt-2 p-2 bg-red-500/5 border border-red-500/10 rounded text-[9px] text-red-400 font-mono">
                                        {step.error}
                                    </div>
                                )}
                                {step.status === "success" && step.output && (
                                    <div className="mt-1 flex items-center gap-1.5 text-[9px] text-emerald-500/40 font-mono italic">
                                        out: {pathBasename(step.output)}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Status and Console */}
                <div className="space-y-6">
                    {/* Overall Progress Orb */}
                    <div className="aspect-square flex items-center justify-center relative">
                        <div className={`absolute inset-0 rounded-full blur-[60px] opacity-20 transition-all duration-1000 ${
                            overallStatus === "running" ? "bg-blue-500" :
                            overallStatus === "success" ? "bg-emerald-500" :
                            overallStatus === "error"   ? "bg-red-500" : "bg-white/5"
                        }`} />
                        <div className="relative z-10 text-center">
                            <div className="text-[48px] font-black text-white/90 leading-none">
                                {Math.round((steps.filter(s => s.status === "success").length / steps.length) * 100)}%
                            </div>
                            <div className="text-[10px] text-white/20 uppercase tracking-[0.3em] mt-3">Pipeline Status</div>
                            <div className={`text-[11px] font-bold mt-1 uppercase tracking-wider ${
                                overallStatus === "running" ? "text-blue-400" :
                                overallStatus === "success" ? "text-emerald-400" :
                                overallStatus === "error"   ? "text-red-400" : "text-white/20"
                            }`}>
                                {overallStatus === "idle" ? "Standby" : overallStatus === "running" ? "Executing..." : 
                                 overallStatus === "success" ? "Finalized" : "Failure"}
                            </div>
                        </div>
                    </div>

                    {/* Console Logs */}
                    <div className="bg-black/40 border border-white/5 rounded-xl p-4 space-y-2">
                        <div className="text-[9px] text-white/15 uppercase tracking-[0.2em] font-bold mb-3 border-b border-white/5 pb-2">
                            Session Output
                        </div>
                        <div className="space-y-1.5 min-h-[100px]">
                            {logs.length === 0 ? (
                                <div className="text-[10px] text-white/5 font-mono italic">Awaiting initialization...</div>
                            ) : (
                                logs.map((log, i) => (
                                    <div key={i} className={`text-[10px] font-mono leading-relaxed ${
                                        log.includes("ERROR") ? "text-red-400" : 
                                        log.includes("completed") ? "text-emerald-400" : "text-white/30"
                                    }`}>
                                        {log}
                                    </div>
                                ))
                            )}
                        </div>
                    </div>

                    {overallStatus === "success" && (
                        <div className="flex gap-2">
                             <Link 
                                href={`/viewer?file=${encodeURIComponent(folderName + '/' + steps[7].output)}`}
                                className="flex-1 flex items-center justify-center gap-2 py-3 bg-emerald-500 text-black rounded-xl text-[11px] font-bold uppercase tracking-widest hover:bg-emerald-400 transition-all hover:scale-[1.02]"
                            >
                                <Eye className="w-3.5 h-3.5" />
                                Review Final Result
                            </Link>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
