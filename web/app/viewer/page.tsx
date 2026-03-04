"use client";

import { useSearchParams } from "next/navigation";
import dynamic from "next/dynamic";
import { Suspense } from "react";
import Link from "next/link";

// Dynamically import NiftiViewer to avoid SSR issues with canvas/window
const NiftiViewer = dynamic(() => import("@/components/NiftiViewer"), { ssr: false });

function ViewerContent() {
    const searchParams = useSearchParams();
    const file = searchParams.get("file");

    if (!file) {
        return <div className="text-white p-8">Error: No file specified.</div>;
    }

    // Construct API URL to serve the file
    // file param is relative path like "study_name/file.nii"
    const fileUrl = `/api/serve-file?path=${encodeURIComponent(file)}`;

    return (
        <div className="flex flex-col h-screen bg-black">
            <div className="h-16 bg-gray-900 border-b border-gray-800 flex items-center px-6 justify-between shrink-0">
                <h1 className="text-white font-bold text-lg truncate max-w-2xl">
                    Viewing: <span className="text-emerald-400 font-mono text-base">{file}</span>
                </h1>
                <Link
                    href="/"
                    className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded text-sm font-medium transition-colors"
                >
                    Close & Return
                </Link>
            </div>

            <div className="flex-grow relative overflow-hidden">
                <NiftiViewer url={fileUrl} />
            </div>
        </div>
    );
}

export default function ViewerPage() {
    return (
        <Suspense fallback={<div className="bg-black h-screen text-white flex items-center justify-center">Loading viewer...</div>}>
            <ViewerContent />
        </Suspense>
    );
}
