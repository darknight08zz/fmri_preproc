"use client";

import { useState } from "react";
import Link from "next/link";

export default function STCPage() {
    const [fileName, setFileName] = useState("");
    const [tr, setTr] = useState<string>("");
    const [slices, setSlices] = useState<string>("");
    const [ta, setTa] = useState<string>("");
    const [sliceOrder, setSliceOrder] = useState<string>("ascending");
    const [refSlice, setRefSlice] = useState<string>("0");
    const [isLoading, setIsLoading] = useState(false);
    const [status, setStatus] = useState<string | null>(null);
    const [result, setResult] = useState<any>(null);
    const [error, setError] = useState<string | null>(null);

    const handleRunSTC = async () => {
        if (!fileName) {
            setError("Please enter a filename.");
            return;
        }

        setIsLoading(true);
        setStatus("Starting STC pipeline...");
        setError(null);
        setResult(null);

        try {
            const response = await fetch("/api/stc", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    filename: fileName.trim(),
                    tr: tr ? parseFloat(tr) : null,
                    slices: slices ? parseInt(slices) : null,
                    ta: ta ? parseFloat(ta) : null,
                    slice_order: sliceOrder,
                    ref_slice: parseInt(refSlice)
                }),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Failed to run STC");
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
            <h1 className="text-3xl font-bold text-white mb-6">Slice Timing Correction</h1>

            <div className="max-w-2xl bg-[#252526] p-6 rounded-lg shadow-lg border border-[#333]">
                <div className="space-y-4 mb-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-400 mb-2">
                            Input NIfTI Filename
                        </label>
                        <input
                            type="text"
                            value={fileName}
                            onChange={(e) => setFileName(e.target.value)}
                            placeholder="e.g., rsub-01_task-rest_bold.nii"
                            className="w-full px-4 py-2 bg-[#3c3c3c] text-white border border-[#555] rounded focus:outline-none focus:border-blue-500"
                        />
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">
                                TR (seconds)
                            </label>
                            <input
                                type="number"
                                step="0.1"
                                value={tr}
                                onChange={(e) => setTr(e.target.value)}
                                placeholder="e.g., 2.0"
                                className="w-full px-4 py-2 bg-[#3c3c3c] text-white border border-[#555] rounded focus:outline-none focus:border-blue-500"
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">
                                Number of Slices
                            </label>
                            <input
                                type="number"
                                value={slices}
                                onChange={(e) => setSlices(e.target.value)}
                                placeholder="e.g., 36"
                                className="w-full px-4 py-2 bg-[#3c3c3c] text-white border border-[#555] rounded focus:outline-none focus:border-blue-500"
                            />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">
                                TA (Acquisition Time)
                            </label>
                            <input
                                type="number"
                                step="0.001"
                                value={ta}
                                onChange={(e) => setTa(e.target.value)}
                                placeholder="e.g., 1.944"
                                className="w-full px-4 py-2 bg-[#3c3c3c] text-white border border-[#555] rounded focus:outline-none focus:border-blue-500"
                            />
                        </div>
                    </div>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">
                                Slice Order
                            </label>
                            <select
                                value={sliceOrder}
                                onChange={(e) => setSliceOrder(e.target.value)}
                                className="w-full px-4 py-2 bg-[#3c3c3c] text-white border border-[#555] rounded focus:outline-none focus:border-blue-500"
                            >
                                <option value="ascending">Ascending (1, 2, 3...)</option>
                                <option value="descending">Descending (N, N-1...)</option>
                                <option value="interleaved">Interleaved (1, 3, 5...)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-400 mb-2">
                                Reference Slice (0-indexed)
                            </label>
                            <input
                                type="number"
                                value={refSlice}
                                onChange={(e) => setRefSlice(e.target.value)}
                                placeholder="0"
                                className="w-full px-4 py-2 bg-[#3c3c3c] text-white border border-[#555] rounded focus:outline-none focus:border-blue-500"
                            />
                        </div>
                    </div>
                </div>

                <button
                    onClick={handleRunSTC}
                    disabled={isLoading}
                    className={`px-6 py-2 rounded font-bold text-white transition-colors ${isLoading
                        ? "bg-gray-600 cursor-not-allowed"
                        : "bg-blue-600 hover:bg-blue-500"
                        }`}
                >
                    {isLoading ? "Processing..." : "Run STC"}
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
                        <h3 className="text-lg font-bold text-green-400">Correction Successful!</h3>

                        <div className="text-sm font-mono bg-[#1e1e1e] p-4 rounded border border-[#333] overflow-x-auto">
                            <p className="text-gray-400 mb-2">Output File:</p>
                            <ul className="space-y-1">
                                <li>Corrected: <span className="text-white">{result.outputs.corrected}</span></li>
                            </ul>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
