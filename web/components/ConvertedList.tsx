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
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden shadow-sm mt-8">
            <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                <h3 className="text-sm font-semibold text-gray-700">Converted NIfTI Files</h3>
            </div>
            <ul className="divide-y divide-gray-100 max-h-60 overflow-y-auto">
                {studies.map((study) => (
                    <li key={study.name} className="px-4 py-3 hover:bg-gray-50 transition-colors flex flex-col gap-3">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-lg bg-indigo-50 text-indigo-600">
                                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                </svg>
                            </div>
                            <div className="flex flex-col">
                                <span className="text-sm text-gray-900 font-bold font-mono">{study.name}</span>
                                <span className="text-xs text-gray-500">{study.fileCount} output files</span>
                            </div>
                        </div>

                        {/* List individual files */}
                        <div className="pl-12 flex flex-col gap-2">
                            {study.files && study.files.length > 0 ? (
                                study.files.map((file) => (
                                    <div key={file} className="flex justify-between items-center bg-gray-50 p-2 rounded border border-gray-100">
                                        <span className="text-xs font-mono text-gray-600 truncate max-w-[200px]" title={file}>
                                            {file}
                                        </span>
                                        <Link
                                            href={`/viewer?file=${encodeURIComponent(study.name + '/' + file)}`}
                                            className="text-xs px-3 py-1 bg-indigo-50 text-indigo-600 rounded hover:bg-indigo-100 font-medium transition-colors border border-indigo-100"
                                        >
                                            View 3D
                                        </Link>
                                    </div>
                                ))
                            ) : (
                                <span className="text-xs text-gray-400 italic">No files found.</span>
                            )}
                        </div>
                    </li>
                ))}
            </ul>
        </div>
    );
}
