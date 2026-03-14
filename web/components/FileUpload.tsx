"use client";

import { useState, ChangeEvent, useEffect } from "react";

interface FileUploadProps {
    onUploadSuccess: () => void;
}

export default function FileUpload({ onUploadSuccess }: FileUploadProps) {
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [folderName, setFolderName] = useState("");
    const [existingFolders, setExistingFolders] = useState<string[]>([]);
    const [isNewStudy, setIsNewStudy] = useState(true);
    const [folderType, setFolderType] = useState<"anat" | "func">("func");

    useEffect(() => {
        const fetchFolders = async () => {
            try {
                const response = await fetch("/api/files");
                if (response.ok) {
                    const data = await response.json();
                    if (data.folders && data.folders.length > 0) {
                        setExistingFolders(data.folders.map((f: any) => f.name));
                        setIsNewStudy(false);
                        setFolderName(data.folders[0].name);
                    }
                }
            } catch (error) {
                console.error("Error fetching folders:", error);
            }
        };

        fetchFolders();
    }, []);

    const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        if (!folderName.trim()) {
            setMessage("Error: Please enter or select a Study Name first.");
            // Clear file input
            e.target.value = "";
            return;
        }

        setUploading(true);
        setMessage(null);

        const formData = new FormData();
        formData.append("folderName", folderName);
        formData.append("folderType", folderType);

        // Append all selected files to formData
        for (let i = 0; i < files.length; i++) {
            formData.append("file", files[i]);
        }

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Upload failed");
            }

            const result = await response.json();
            setMessage(`Successfully uploaded ${result.count} files to "${result.folder}/${result.type}"!`);
            
            // If it was a new study, add it to the dropdown list
            if (isNewStudy && !existingFolders.includes(result.folder)) {
                setExistingFolders(prev => [...prev, result.folder]);
                setIsNewStudy(false);
                setFolderName(result.folder);
            }
            
            onUploadSuccess();
        } catch (error) {
            console.error(error);
            setMessage("Error uploading files.");
        } finally {
            setUploading(false);
            // Clear the input to allow re-uploading same files if needed
            e.target.value = "";
        }
    };

    return (
        <div className="group relative border border-white/[0.08] rounded-xl bg-white/[0.01] overflow-hidden transition-all duration-300 hover:border-emerald-500/30">
            <div className="p-6">
                <div className="flex items-center justify-between mb-5">
                    <h3 className="text-xs font-bold text-white/40 uppercase tracking-widest">Upload DICOM Series</h3>
                    {uploading && (
                        <div className="flex items-center gap-2 text-[10px] text-emerald-400 animate-pulse font-mono">
                            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                            ORGANIZING DATA...
                        </div>
                    )}
                </div>

                <div className="grid grid-cols-1 gap-5">
                    {/* Identification Row */}
                    <div className="flex flex-col sm:flex-row gap-5">
                        <div className="flex-1">
                            <div className="flex items-center justify-between mb-2">
                                <label className="text-[10px] text-white/20 uppercase tracking-tighter">Study Name</label>
                                <div className="flex items-center gap-3">
                                    <button 
                                        onClick={() => setIsNewStudy(false)}
                                        className={`text-[9px] uppercase tracking-wider font-bold transition-colors ${!isNewStudy ? "text-emerald-400" : "text-white/10 hover:text-white/30"}`}
                                    >
                                        [ Existing ]
                                    </button>
                                    <button 
                                        onClick={() => { setIsNewStudy(true); setFolderName(""); }}
                                        className={`text-[9px] uppercase tracking-wider font-bold transition-colors ${isNewStudy ? "text-emerald-400" : "text-white/10 hover:text-white/30"}`}
                                    >
                                        [ New ]
                                    </button>
                                </div>
                            </div>
                            
                            {!isNewStudy && existingFolders.length > 0 ? (
                                <select
                                    value={folderName}
                                    onChange={(e) => setFolderName(e.target.value)}
                                    className="w-full bg-[#0d0e11] border border-white/[0.05] rounded-lg px-3 py-2 text-xs font-mono text-white/70 focus:outline-none focus:border-emerald-500/50"
                                >
                                    {existingFolders.map(name => (
                                        <option key={name} value={name}>{name}</option>
                                    ))}
                                </select>
                            ) : (
                                <input
                                    type="text"
                                    value={folderName}
                                    onChange={(e) => setFolderName(e.target.value)}
                                    placeholder="e.g. Patient_01"
                                    className="w-full bg-[#0d0e11] border border-white/[0.05] rounded-lg px-3 py-2 text-xs font-mono text-white/70 placeholder:text-white/5 focus:outline-none focus:border-emerald-500/50"
                                />
                            )}
                        </div>

                        <div className="w-full sm:w-40">
                            <label className="block text-[10px] text-white/20 uppercase tracking-tighter mb-2">Scan Type</label>
                            <select
                                value={folderType}
                                onChange={(e) => setFolderType(e.target.value as "anat" | "func")}
                                className="w-full bg-[#0d0e11] border border-white/[0.05] rounded-lg px-3 py-2 text-xs font-mono text-white/70 focus:outline-none focus:border-emerald-500/50"
                            >
                                <option value="anat">Anat (T1w)</option>
                                <option value="func">Func (BOLD)</option>
                            </select>
                        </div>
                    </div>

                    {/* Drop Zone */}
                    <div className="relative">
                        <label className={`
                            relative flex flex-col items-center justify-center py-10 px-4 
                            border-2 border-dashed rounded-xl transition-all duration-300
                            ${!folderName.trim() || uploading ? "border-white/[0.03] opacity-40 cursor-not-allowed" : "border-white/[0.08] hover:border-emerald-500/30 cursor-pointer bg-white/[0.01] hover:bg-emerald-500/[0.02]"}
                        `}>
                            <input
                                type="file"
                                multiple
                                onChange={handleFileChange}
                                disabled={uploading || !folderName.trim()}
                                className="hidden"
                            />
                            
                            <div className="mb-3 text-white/20">
                                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                                    <path strokeLinecap="round" strokeLinejoin="round" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                                </svg>
                            </div>

                            <span className="text-[11px] text-white/40 text-center leading-relaxed">
                                Drop DICOM files here<br />
                                <span className="text-[9px] text-white/10 uppercase tracking-widest font-mono">or click to browse</span>
                            </span>

                            {!folderName.trim() && !uploading && (
                                <div className="absolute inset-0 flex items-center justify-center bg-[#0a0a0a]/80 backdrop-blur-[2px] rounded-xl">
                                    <span className="text-[9px] font-bold text-emerald-500/60 uppercase tracking-widest">Set Study Name First</span>
                                </div>
                            )}
                        </label>
                    </div>

                    {message && (
                        <div className={`
                            text-[10px] font-mono px-3 py-2 rounded border transition-all duration-300
                            ${message.includes("Error") 
                                ? "bg-red-500/5 text-red-400 border-red-500/20" 
                                : "bg-emerald-500/5 text-emerald-400 border-emerald-500/20"}
                        `}>
                            {message.toUpperCase()}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

