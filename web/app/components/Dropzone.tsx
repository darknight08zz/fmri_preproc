
"use client";

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, File as FileIcon, X, AlertCircle, Loader2 } from 'lucide-react';
import axios from 'axios';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

interface DropzoneProps {
    onUploadComplete: (path: string) => void;
    onCancel: () => void;
}

export default function Dropzone({ onUploadComplete, onCancel }: DropzoneProps) {
    const [mode, setMode] = useState<'file' | 'url'>('file');
    const [url, setUrl] = useState('');

    const [files, setFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);
    const [progress, setProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);

    const onDrop = useCallback((acceptedFiles: File[]) => {
        setFiles(prev => [...prev, ...acceptedFiles]);
        setError(null);
    }, []);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

    const removeFile = (name: string) => {
        setFiles(files.filter(f => f.name !== name));
    };

    const handleUpload = async () => {
        setUploading(true);
        setProgress(0);
        setError(null);

        try {
            if (mode === 'file') {
                if (files.length === 0) return;

                const formData = new FormData();
                files.forEach(file => {
                    formData.append('files', file);
                });

                const res = await axios.post('http://localhost:8000/datasets/upload', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                    onUploadProgress: (progressEvent) => {
                        if (progressEvent.total) {
                            const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                            setProgress(percent);
                        }
                    }
                });
                if (res.data.path) onUploadComplete(res.data.path);

            } else {
                // URL Mode
                if (!url) return;
                // Fake progress for URL since we can't track server-side download easily without websockets
                const timer = setInterval(() => setProgress(prev => Math.min(prev + 10, 90)), 500);

                const res = await axios.post('http://localhost:8000/datasets/upload-url', { url });

                clearInterval(timer);
                setProgress(100);

                if (res.data.path) onUploadComplete(res.data.path);
            }
        } catch (err) {
            console.error(err);
            if (axios.isAxiosError(err) && err.response) {
                setError(err.response.data.detail || "Upload failed. Check console.");
            } else {
                setError("Upload failed. Check console.");
            }
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="w-full max-w-2xl mx-auto">
            <div className="flex justify-between items-center mb-6">
                <h3 className="text-xl font-bold text-white">Import Data</h3>
                <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-700">
                    <button
                        onClick={() => setMode('file')}
                        className={cn("px-4 py-1.5 rounded-md text-sm font-medium transition-all", mode === 'file' ? "bg-slate-700 text-white shadow-sm" : "text-slate-400 hover:text-white")}
                    >
                        File Upload
                    </button>
                    <button
                        onClick={() => setMode('url')}
                        className={cn("px-4 py-1.5 rounded-md text-sm font-medium transition-all", mode === 'url' ? "bg-slate-700 text-white shadow-sm" : "text-slate-400 hover:text-white")}
                    >
                        From URL
                    </button>
                </div>
            </div>

            {mode === 'file' ? (
                <>
                    {/* Drop Area */}
                    <div
                        {...getRootProps()}
                        className={cn(
                            "border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all",
                            isDragActive ? "border-cyan-500 bg-cyan-900/20" : "border-slate-700 hover:border-slate-500 hover:bg-slate-800/50",
                            uploading ? "opacity-50 pointer-events-none" : ""
                        )}
                    >
                        <input {...getInputProps()} />
                        <UploadCloud className={cn("w-12 h-12 mx-auto mb-4", isDragActive ? "text-cyan-400" : "text-slate-500")} />
                        {isDragActive ? (
                            <p className="text-cyan-400 font-medium">Drop the files here ...</p>
                        ) : (
                            <div className="space-y-2">
                                <p className="text-slate-300 font-medium">Drag &apos;n&apos; drop files here, or click to select files</p>
                                <p className="text-sm text-slate-500">Supports .nii.gz, .json, .zip</p>
                            </div>
                        )}
                    </div>

                    {/* File List */}
                    {files.length > 0 && (
                        <div className="mt-6 space-y-3">
                            <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Selected Files ({files.length})</h4>
                            <div className="max-h-60 overflow-y-auto space-y-2 custom-scrollbar pr-2">
                                {files.map((file, idx) => (
                                    <div key={`${file.name}-${idx}`} className="flex items-center justify-between p-3 bg-slate-900 rounded-lg border border-slate-800">
                                        <div className="flex items-center gap-3 overflow-hidden">
                                            <FileIcon className="w-5 h-5 text-blue-400 flex-shrink-0" />
                                            <span className="text-sm text-slate-300 truncate">{file.name}</span>
                                            <span className="text-xs text-slate-500">{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                                        </div>
                                        {!uploading && (
                                            <button onClick={() => removeFile(file.name)} className="text-slate-500 hover:text-red-400">
                                                <X className="w-4 h-4" />
                                            </button>
                                        )}
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </>
            ) : (
                <div className="bg-slate-900 p-6 rounded-xl border border-slate-800">
                    <label className="block text-sm text-slate-400 mb-2">Dataset URL</label>
                    <input
                        type="text"
                        value={url}
                        onChange={(e) => setUrl(e.target.value)}
                        placeholder="https://openneuro.org/..."
                        className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-3 text-white focus:ring-2 focus:ring-cyan-500 outline-none"
                    />
                    <p className="text-xs text-slate-500 mt-2">
                        Enter a direct link to a file (e.g. .zip, .tar.gz, .nii.gz). Large files may take time to process.
                    </p>
                </div>
            )}

            {/* Progress & Error */}
            {uploading && (
                <div className="mt-6 space-y-2">
                    <div className="flex justify-between text-sm text-slate-400">
                        <span>{mode === 'url' ? 'Downloading from URL...' : 'Uploading...'}</span>
                        <span>{progress}%</span>
                    </div>
                    <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-cyan-500 transition-all duration-300" style={{ width: `${progress}%` }}></div>
                    </div>
                </div>
            )}

            {error && (
                <div className="mt-4 p-4 bg-red-950/30 border border-red-900/50 rounded-lg flex items-center gap-3">
                    <AlertCircle className="w-5 h-5 text-red-400" />
                    <span className="text-red-200 text-sm">{error}</span>
                </div>
            )}

            {/* Actions */}
            <div className="flex gap-4 mt-8 justify-end">
                <button
                    onClick={onCancel}
                    disabled={uploading}
                    className="px-4 py-2 text-slate-400 hover:text-white transition-colors disabled:opacity-50"
                >
                    Cancel
                </button>
                <button
                    onClick={handleUpload}
                    disabled={(mode === 'file' ? files.length === 0 : !url) || uploading}
                    className="px-6 py-2 bg-gradient-to-r from-blue-600 to-cyan-600 rounded-lg text-white font-medium hover:from-blue-500 hover:to-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                >
                    {uploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <UploadCloud className="w-4 h-4" />}
                    {mode === 'url' ? 'Download' : 'Upload Files'}
                </button>
            </div>
        </div>
    );
}
