"use client";

import { useState, ChangeEvent } from "react";

interface FileUploadProps {
    onUploadSuccess: () => void;
}

export default function FileUpload({ onUploadSuccess }: FileUploadProps) {
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [folderName, setFolderName] = useState("");

    const handleFileChange = async (e: ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (!files || files.length === 0) return;

        if (!folderName.trim()) {
            setMessage("Error: Please enter a Study Name first.");
            // Clear file input
            e.target.value = "";
            return;
        }

        setUploading(true);
        setMessage(null);

        const formData = new FormData();
        formData.append("folderName", folderName);

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
            setMessage(`Successfully uploaded ${result.count} files to "${result.folder}"!`);
            // Clear folder name for next upload? Or keep it? Keeping it might be useful for adding more files.
            // setFolderName(""); 
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
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
            <h3 className="text-lg font-bold text-gray-900 mb-4">Upload DICOM Series</h3>

            <div className="flex flex-col gap-4">
                <div>
                    <label htmlFor="folderName" className="block text-sm font-medium text-gray-700 mb-1">
                        Study Name / ID <span className="text-red-500">*</span>
                    </label>
                    <input
                        type="text"
                        id="folderName"
                        value={folderName}
                        onChange={(e) => setFolderName(e.target.value)}
                        placeholder="e.g. Patient_001_fMRI"
                        className="block w-full rounded-md border-gray-300 shadow-sm focus:border-emerald-500 focus:ring-emerald-500 sm:text-sm px-3 py-2 border"
                    />
                    <p className="mt-1 text-xs text-gray-500">
                        Enter a unique name for this study folder.
                    </p>
                </div>

                <div>
                    <label className="block">
                        <span className="sr-only">Choose files</span>
                        <input
                            type="file"
                            multiple
                            onChange={handleFileChange}
                            disabled={uploading || !folderName.trim()}
                            className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-full file:border-0
                file:text-sm file:font-semibold
                file:bg-emerald-50 file:text-emerald-700
                hover:file:bg-emerald-100
                cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                        />
                    </label>
                </div>

                {uploading && (
                    <div className="text-sm text-blue-600 animate-pulse">Uploading and organizing files...</div>
                )}

                {message && (
                    <div className={`text-sm ${message.includes("Error") ? "text-red-600" : "text-emerald-600"}`}>
                        {message}
                    </div>
                )}
            </div>
        </div>
    );
}
