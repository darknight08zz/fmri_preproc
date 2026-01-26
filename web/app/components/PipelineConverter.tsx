
import React, { useState, useEffect } from 'react';
import { Play, CheckCircle, Loader2, X, Server } from 'lucide-react';
import NiftiViewer from './NiftiViewer';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: (string | undefined | null | false)[]) {
    return twMerge(clsx(inputs));
}

interface PipelineConverterProps {
    inputDir: string;
    subject: string;
    onClose: () => void;
}

interface PipelineStatus {
    status: 'idle' | 'running' | 'completed' | 'failed' | 'unknown';
    stage?: string;
    percent?: number;
    message?: string;
    output_path?: string;
    error?: string;
}

const STAGES = [
    "Stage 1: Input Ingestion - Reading Headers",
    "Stage 2: Series Identification - Grouping",
    "Stage 3: Metadata Extraction",
    "Stage 4: Slice Sorting (Z-Axis)",
    "Stage 5: Time Sorting",
    "Stage 6 & 7: Building 4D Volume",
    "Stage 8: Affine Matrix Construction",
    "Stage 9: Slice Timing Calculation",
    "Stage 10: Writing NIfTI File",
    "Stage 11: Verification"
];

export default function PipelineConverter({ inputDir, subject, onClose }: PipelineConverterProps) {
    const [pipelineState, setPipelineState] = useState<PipelineStatus>({ status: 'idle' });
    const [logs, setLogs] = useState<string[]>([]);

    // Polling effect
    useEffect(() => {
        let interval: NodeJS.Timeout;

        if (pipelineState.status === 'running') {
            interval = setInterval(async () => {
                try {
                    const res = await fetch(`http://localhost:8000/convert/status/${subject}`);
                    const data = await res.json();
                    if (data.status !== 'unknown') {
                        setPipelineState(prev => {
                            // simple log append if stage changed
                            if (data.stage !== prev.stage) {
                                setLogs(l => [...l, `[${new Date().toLocaleTimeString()}] ${data.stage} (${data.percent}%)`]);
                            }
                            return data;
                        });
                    }
                } catch {
                    // Fail silently on poll error
                }
            }, 1000);
        }

        return () => clearInterval(interval);
    }, [pipelineState.status, subject]);

    const startPipeline = async () => {
        try {
            setPipelineState({ status: 'running', percent: 0, stage: 'Starting...' });
            const res = await fetch('http://localhost:8000/convert/dicom', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input_dir: inputDir,
                    subject: subject,
                    use_pipeline: true
                }),
            });
            const data = await res.json();
            if (!res.ok) {
                setPipelineState({ status: 'failed', error: data.detail || "Failed to start" });
                setLogs(l => [...l, `Error: ${data.detail}`]);
            } else {
                setLogs(l => [...l, `Pipeline started for ${subject}`]);
            }
        } catch {
            setPipelineState({ status: 'failed', error: "Network error" });
        }
    };

    const currentStageIndex = STAGES.findIndex(s => pipelineState.stage?.startsWith(s.split(':')[0])) ?? -1;
    // If completed, all done
    const isComplete = pipelineState.status === 'completed';

    // Construct viewer URL if complete
    let viewerUrl = '';
    if (isComplete && pipelineState.output_path) {
        // Path logic: d:\...\converted_data\sub-01\func\file.nii.gz -> /files/sub-01/func/file.nii.gz
        // We assume backend mounted 'converted_data' to '/files'
        const parts = pipelineState.output_path.split('converted_data');
        if (parts.length > 1) {
            const relativePath = parts[1].replace(/\\/g, '/'); // Ensure forward slashes for URL
            viewerUrl = `http://localhost:8000/files${relativePath}`;
        }
    }

    return (
        <div className="fixed inset-0 bg-black/90 flex items-center justify-center z-50 p-6 backdrop-blur-sm">
            <div className="bg-slate-950 w-full max-w-6xl h-[90vh] rounded-2xl border border-slate-700 shadow-2xl flex flex-col overflow-hidden relative">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors z-50"
                >
                    <X className="w-6 h-6" />
                </button>

                <header className="p-6 border-b border-slate-800 bg-slate-900/50">
                    <h2 className="text-2xl font-bold flex items-center gap-3 text-cyan-400">
                        <Server className="w-6 h-6" />
                        DICOM to NIfTI Pipeline
                    </h2>
                    <p className="text-slate-400 mt-1 font-mono text-sm">Target: {subject} | Source: {inputDir}</p>
                </header>

                <div className="flex-1 overflow-hidden grid grid-cols-1 lg:grid-cols-3">
                    {/* Left: Visualizer Stages */}
                    <div className="p-6 overflow-y-auto border-r border-slate-800 custom-scrollbar bg-slate-900/20 lg:col-span-1">
                        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-500 mb-4">Pipeline Stages</h3>
                        <div className="space-y-4">
                            {STAGES.map((stage, idx) => {
                                const isActive = pipelineState.status === 'running' && (pipelineState.stage?.startsWith(stage.split(':')[0]) || currentStageIndex === idx);
                                const isDone = isComplete || (currentStageIndex > idx) || (pipelineState.percent === 100);

                                return (
                                    <div key={idx} className={cn(
                                        "p-3 rounded-lg border transition-all duration-300",
                                        isActive ? "bg-cyan-950/30 border-cyan-500/50 shadow-[0_0_15px_rgba(6,182,212,0.15)]" :
                                            isDone ? "bg-green-950/10 border-green-900/30 opacity-70" : "bg-slate-900 border-slate-800 opacity-40"
                                    )}>
                                        <div className="flex items-center gap-3">
                                            {isActive ? (
                                                <Loader2 className="w-5 h-5 text-cyan-400 animate-spin" />
                                            ) : isDone ? (
                                                <CheckCircle className="w-5 h-5 text-green-500" />
                                            ) : (
                                                <div className="w-5 h-5 rounded-full border border-slate-600" />
                                            )}
                                            <span className={cn("font-medium text-sm", isActive && "text-cyan-300", isDone && "text-green-300")}>
                                                {stage}
                                            </span>
                                        </div>
                                        {isActive && (
                                            <div className="mt-2 h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-cyan-500 transition-all duration-500"
                                                    style={{ width: `${pipelineState.percent}%` }}
                                                />
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>

                    {/* Right: Logs & Actions / Viewer */}
                    <div className="p-0 flex flex-col bg-slate-950 lg:col-span-2">
                        {isComplete && viewerUrl ? (
                            <div className="flex-1 flex flex-col h-full bg-black relative">
                                <NiftiViewer
                                    url={viewerUrl}
                                    className="w-full h-full"
                                />
                                <div className="absolute bottom-4 right-4 z-50">
                                    <button
                                        onClick={onClose}
                                        className="bg-slate-800 hover:bg-slate-700 text-white px-6 py-2 rounded-xl font-medium transition-colors border border-slate-600 shadow-xl"
                                    >
                                        Close Viewer
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div className="p-6 flex flex-col h-full">
                                <div className="flex-1 bg-black rounded-lg border border-slate-800 p-4 font-mono text-xs text-green-400 overflow-y-auto mb-6 custom-scrollbar">
                                    {logs.length === 0 && <span className="text-slate-600 opacity-50">System ready. Waiting to start...</span>}
                                    {logs.map((log, i) => (
                                        <div key={i} className="mb-1">{log}</div>
                                    ))}
                                    {pipelineState.status === 'running' && (
                                        <div className="animate-pulse">_</div>
                                    )}
                                </div>

                                {pipelineState.status === 'idle' ? (
                                    <button
                                        onClick={startPipeline}
                                        className="w-full bg-cyan-600 hover:bg-cyan-500 text-white py-4 rounded-xl font-bold text-lg transition-transform active:scale-[0.98] shadow-lg shadow-cyan-900/20 flex items-center justify-center gap-2"
                                    >
                                        <Play className="w-5 h-5" /> Start Pipeline
                                    </button>
                                ) : pipelineState.status === 'failed' ? (
                                    <div className="p-4 bg-red-950/30 border border-red-500/30 rounded-xl text-center">
                                        <h4 className="text-red-400 font-bold mb-1">Conversion Failed</h4>
                                        <p className="text-red-300/70 text-sm">{pipelineState.error}</p>
                                        <button
                                            onClick={() => setPipelineState({ status: 'idle' })}
                                            className="mt-4 bg-red-900/50 hover:bg-red-800/50 text-white px-4 py-2 rounded-lg text-sm"
                                        >
                                            Retry
                                        </button>
                                    </div>
                                ) : (
                                    <div className="text-center text-slate-500 text-sm animate-pulse">Running pipeline...</div>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
