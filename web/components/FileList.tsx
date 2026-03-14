"use client";

import { useEffect, useState } from "react";

interface FileListProps {
    refreshTrigger: number;
    onConversionSuccess: () => void;
}

interface StudyFolder {
    name: string;
    anatCount: number;
    funcCount: number;
}

export default function FileList({ refreshTrigger, onConversionSuccess }: FileListProps) {
    const [folders, setFolders] = useState<StudyFolder[]>([]);
    const [loading, setLoading] = useState(true);
    // converting key: "folderName-anat" or "folderName-func"
    const [converting, setConverting] = useState<Record<string, boolean>>({});
    const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);

    useEffect(() => {
        const fetchFolders = async () => {
            try {
                const response = await fetch("/api/files");
                if (response.ok) {
                    const data = await response.json();
                    setFolders(data.folders || []);
                }
            } catch (error) {
                console.error("Error fetching folders:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchFolders();
    }, [refreshTrigger]);

    const handleConvert = async (folderName: string, type: "anat" | "func") => {
        const key = `${folderName}-${type}`;
        setConverting(prev => ({ ...prev, [key]: true }));
        setMessage(null);

        try {
            const response = await fetch("/api/convert", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ folderName, type }),
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Conversion failed");
            }

            setMessage({ type: 'success', text: `Successfully converted "${folderName}/${type}"!` });
            onConversionSuccess();

        } catch (error: any) {
            console.error("Conversion error:", error);
            setMessage({ type: 'error', text: error.message || "Failed to convert study." });
        } finally {
            setConverting(prev => ({ ...prev, [key]: false }));
        }
    };

    if (loading) {
        return <div className="text-gray-500 text-sm animate-pulse">Loading studies...</div>;
    }

    if (folders.length === 0) {
        return <div className="text-gray-400 text-sm italic">No studies uploaded yet.</div>;
    }

    const renderConvertButton = (folderName: string, type: "anat" | "func", count: number) => {
        if (count === 0) return null;
        
        const key = `${folderName}-${type}`;
        const isConverting = converting[key];

        return (
            <div className="flex items-center justify-between py-2 border-t border-white/[0.04] mt-2 group/row">
                <div className="flex items-center gap-3">
                    <span className={`text-[11px] font-bold uppercase tracking-widest px-1.5 py-0.5 rounded border transition-colors ${
                        type === 'anat' ? 'text-blue-400 bg-blue-500/10 border-blue-500/20' : 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                    }`}>
                        {type}
                    </span>
                    <span className="text-[12px] text-white/20 font-mono tracking-tighter">{count} DICOM Slices</span>
                </div>
                <button
                    onClick={() => handleConvert(folderName, type)}
                    disabled={isConverting}
                    className={`
                        text-[12px] px-3 py-1.5 rounded font-bold uppercase tracking-wider transition-all duration-200 border
                        ${isConverting
                            ? 'bg-white/5 text-white/20 border-white/5 cursor-wait'
                            : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/10 hover:bg-emerald-500/20 hover:border-emerald-500/30'
                        }
                    `}
                >
                    {isConverting ? (
                        <span className="flex items-center gap-1.5">
                            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            ORGANIZING...
                        </span>
                    ) : (
                        `Convert ${type}`
                    )}
                </button>
            </div>
        );
    };

    return (
        <div className="bg-white/[0.01] rounded-xl border border-white/[0.08] overflow-hidden">
            <div className="px-4 py-3 bg-white/[0.02] border-b border-white/[0.08] flex justify-between items-center">
                <h3 className="text-[13px] font-bold text-white/40 uppercase tracking-widest">Uploaded Studies</h3>
                {message && (
                    <span className={`text-[11px] font-bold px-2 py-1 rounded transition-all duration-300 ${message.type === 'success' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-red-500/10 text-red-400'}`}>
                        {message.text.toUpperCase()}
                    </span>
                )}
            </div>
            
            {folders.length === 0 ? (
                <div className="p-8 text-center">
                    <span className="text-[10px] text-white/10 uppercase tracking-[.2em] font-mono">No data found</span>
                </div>
            ) : (
                <ul className="divide-y divide-white/[0.04] max-h-[320px] overflow-y-auto no-scrollbar">
                    {folders.map((folder) => (
                        <li key={folder.name} className="px-4 py-4 hover:bg-white/[0.02] transition-all duration-200 group flex flex-col">
                            <div className="flex items-center gap-3">
                                <div className="p-2 rounded-lg bg-white/[0.03] text-white/20 group-hover:text-emerald-500 group-hover:bg-emerald-500/10 transition-all duration-300">
                                    <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                                    </svg>
                                </div>
                                <span className="text-[14px] text-white/70 font-bold font-mono tracking-tight group-hover:text-white transition-colors">
                                    {folder.name}
                                </span>
                            </div>
                            
                            <div className="pl-11 mt-1">
                                {renderConvertButton(folder.name, 'anat', folder.anatCount)}
                                {renderConvertButton(folder.name, 'func', folder.funcCount)}
                            </div>
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}

