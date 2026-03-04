"use client";

import { useEffect, useState } from "react";

interface FileListProps {
    refreshTrigger: number;
    onConversionSuccess: () => void;
}

interface StudyFolder {
    name: string;
    fileCount: number;
}

export default function FileList({ refreshTrigger, onConversionSuccess }: FileListProps) {
    const [folders, setFolders] = useState<StudyFolder[]>([]);
    const [loading, setLoading] = useState(true);
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

    const handleConvert = async (folderName: string) => {
        setConverting(prev => ({ ...prev, [folderName]: true }));
        setMessage(null);

        try {
            const response = await fetch("/api/convert", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ folderName }),
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.error || "Conversion failed");
            }

            setMessage({ type: 'success', text: `Successfully converted "${folderName}"!` });
            onConversionSuccess();

        } catch (error: any) {
            console.error("Conversion error:", error);
            setMessage({ type: 'error', text: error.message || "Failed to convert study." });
        } finally {
            setConverting(prev => ({ ...prev, [folderName]: false }));
        }
    };

    if (loading) {
        return <div className="text-gray-500 text-sm animate-pulse">Loading studies...</div>;
    }

    if (folders.length === 0) {
        return <div className="text-gray-400 text-sm italic">No studies uploaded yet.</div>;
    }

    return (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm">
            <div className="px-4 py-3 bg-gray-50 border-b border-gray-200 flex justify-between items-center">
                <h3 className="text-sm font-semibold text-gray-700">Uploaded Studies</h3>
                {message && (
                    <span className={`text-xs font-medium px-2 py-1 rounded ${message.type === 'success' ? 'bg-emerald-100 text-emerald-700' : 'bg-red-100 text-red-700'
                        }`}>
                        {message.text}
                    </span>
                )}
            </div>
            <ul className="divide-y divide-gray-100 max-h-60 overflow-y-auto">
                {folders.map((folder) => (
                    <li key={folder.name} className="px-4 py-3 hover:bg-gray-50 transition-colors flex items-center justify-between group">
                        <div className="flex items-center gap-3">
                            <svg className="w-5 h-5 text-gray-400 group-hover:text-emerald-500 transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                            </svg>
                            <div className="flex flex-col">
                                <span className="text-sm text-gray-700 font-medium font-mono">{folder.name}</span>
                                <span className="text-xs text-gray-400">{folder.fileCount} files</span>
                            </div>
                        </div>

                        <button
                            onClick={() => handleConvert(folder.name)}
                            disabled={converting[folder.name]}
                            className={`text-xs px-3 py-1.5 rounded-md font-medium transition-all
                    ${converting[folder.name]
                                    ? 'bg-gray-100 text-gray-400 cursor-wait'
                                    : 'bg-emerald-50 text-emerald-600 hover:bg-emerald-100 border border-emerald-200 hover:shadow-sm'
                                }`}
                        >
                            {converting[folder.name] ? (
                                <span className="flex items-center gap-1">
                                    <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Converting...
                                </span>
                            ) : (
                                "Convert to NIfTI"
                            )}
                        </button>
                    </li>
                ))}
            </ul>
        </div>
    );
}
