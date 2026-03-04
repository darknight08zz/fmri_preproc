
"use client";

import { useState } from "react";
import Link from "next/link";

export default function RealignPage() {
    const [fileName, setFileName] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<string | null>(null);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const handleRunRealign = async () => {
        if (!fileName) {
            setError("Please enter a filename.");
            return;
        }

        setIsLoading(true);
        setStatus("Starting pipeline...");
        setError(null);
        setResult(null);

        try {
            const response = await fetch("/api/realign", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ filename: fileName.trim() }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Failed to run realignment");
            }

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
        <div className="min-h-screen bg-[#1e1e1e] text-[#d4d4d4] p-8 font-sans">
            <div className="mt-8">
                <Link href="/" className="text-blue-400 hover:underline">
                    &larr; Back to Dashboard
                </Link>
            </div>
            <h1 className="text-3xl font-bold text-white mb-6">Realign: Estimate & Reslice</h1>

            <div className="max-w-2xl bg-[#252526] p-6 rounded-lg shadow-lg border border-[#333]">
                <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-400 mb-2">
                        Input NIfTI Filename (must be in `check-convert` folder or absolute path)
                    </label>
                    <input
                        type="text"
                        value={fileName}
                        onChange={(e) => setFileName(e.target.value)}
                        placeholder="e.g., sub-01_task-rest_bold.nii"
                        className="w-full px-4 py-2 bg-[#3c3c3c] text-white border border-[#555] rounded focus:outline-none focus:border-blue-500"
                    />
                </div>

                <button
                    onClick={handleRunRealign}
                    disabled={isLoading}
                    className={`px-6 py-2 rounded font-bold text-white transition-colors ${isLoading
                        ? "bg-gray-600 cursor-not-allowed"
                        : "bg-blue-600 hover:bg-blue-500"
                        }`}
                >
                    {isLoading ? "Processing..." : "Run Realign"}
                </button>

                {status && (
                    <div className="mt-4 p-4 bg-[#1e1e1e] rounded border border-[#333]">
                        <p className="text-gray-300">Status: <span className="font-mono text-cyan-400">{status}</span></p>
                    </div>
                )}

                {error && (
                    <div className="mt-4 p-4 bg-red-900/20 border border-red-800 rounded text-red-400">
                        Error: {error}
                    </div>
                )}

                {result && result.outputs && (
                    <div className="mt-6 space-y-4">
                        <h3 className="text-lg font-bold text-green-400">Realign Successful!</h3>

                        <div className="text-sm font-mono bg-[#1e1e1e] p-4 rounded border border-[#333] overflow-x-auto">
                            <p className="text-gray-400 mb-2">Output Files:</p>
                            <ul className="space-y-1">
                                <li>Resliced: <span className="text-white">{result.outputs.resliced}</span></li>
                                <li>Mean: <span className="text-white">{result.outputs.mean}</span></li>
                                <li>Parameters: <span className="text-white">{result.outputs.motion_params}</span></li>
                            </ul>
                        </div>

                        {/* Motion Plot Preview is tricky since it's on local disk. 
                            We haven't built a static file server for outputs yet.
                            For now, just listing the path is fine. */}
                    </div>
                )}
            </div>

            
        </div>
    );
}
