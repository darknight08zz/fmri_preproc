"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

interface ConvertedListProps {
    refreshTrigger: number;
}

interface ConvertedStudy {
    name: string;
    fileCount: number;
    files: string[];
}

export default function ConvertedList({ refreshTrigger }: ConvertedListProps) {
    const [studies, setStudies] = useState<ConvertedStudy[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchConverted = async () => {
            try {
                const response = await fetch("/api/converted-files");
                if (response.ok) {
                    const data = await response.json();
                    setStudies(data.folders || []);
                }
            } catch (error) {
                console.error("Error fetching converted studies:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchConverted();
    }, [refreshTrigger]);

    if (loading) {
        return <div className="text-gray-500 text-sm animate-pulse">Loading converted data...</div>;
    }

    if (studies.length === 0) {
        return <div className="text-gray-400 text-sm italic">No converted files yet.</div>;
    }

    return (
        <div className="bg-white/[0.01] rounded-xl border border-white/[0.08] overflow-hidden mt-8">
            <div className="px-4 py-3 bg-white/[0.02] border-b border-white/[0.08]">
                <h3 className="text-[13px] font-bold text-white/40 uppercase tracking-widest">Converted NIfTI Files</h3>
            </div>
            
            {studies.length === 0 ? (
                <div className="p-8 text-center text-[10px] text-white/10 uppercase tracking-[.2em] font-mono">
                    No converted data
                </div>
            ) : (
                <ul className="divide-y divide-white/[0.04] max-h-[240px] overflow-y-auto no-scrollbar">
                    {studies.map((study) => (
                        <li key={study.name} className="px-4 py-4 hover:bg-white/[0.02] transition-all duration-200 flex flex-col gap-4 group">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
                                    <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                </div>
                                <div className="flex flex-col">
                                    <span className="text-[14px] text-white/70 font-bold font-mono group-hover:text-white transition-colors">{study.name}</span>
                                    <span className="text-[12px] text-white/20 uppercase tracking-wider">{study.fileCount} output volumes</span>
                                </div>
                            </div>

                            <div className="pl-11 flex flex-col gap-2">
                                {study.files && study.files.length > 0 ? (
                                    study.files.map((file) => (
                                        <div key={file} className="flex justify-between items-center bg-white/[0.02] p-2 rounded border border-white/[0.04] hover:border-white/[0.1] transition-all">
                                            <span className="text-[13px] font-mono text-white/40 break-all truncate mr-4" title={file}>
                                                {file}
                                            </span>
                                            <Link
                                                href={`/viewer?file=${encodeURIComponent(study.name + '/' + file)}`}
                                                className="text-[12px] px-3 py-1 bg-emerald-500/10 text-emerald-400 rounded hover:bg-emerald-500/20 font-bold uppercase tracking-wider transition-all border border-emerald-500/10"
                                            >
                                                Viewer
                                            </Link>
                                        </div>
                                    ))
                                ) : (
                                    <span className="text-[10px] text-white/10 italic">No files available</span>
                                )}
                            </div>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

