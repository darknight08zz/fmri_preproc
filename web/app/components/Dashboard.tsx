"use client";

import React, { useState } from 'react';
import { AlertCircle, CheckCircle, Activity, Play } from 'lucide-react';
import Dropzone from './Dropzone';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface ValidationIssue {
  severity: 'critical' | 'warning';
  message: string;
  scan_path: string;
}

interface ValidationReport {
  valid: boolean;
  issues: ValidationIssue[];
  pipeline_config: Record<string, boolean>;
}

export default function Dashboard() {
  const [selectedSubject, setSelectedSubject] = useState<string | null>(null);
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);

  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [uploadedPath, setUploadedPath] = useState<string>("");
  const [availableUploads, setAvailableUploads] = useState<{ name: string, path: string }[]>([]); // New state

  // Fetch available uploads and restore last session
  const fetchUploads = () => {
    fetch('http://localhost:8000/datasets/uploads')
      .then(res => res.json())
      .then(data => {
        setAvailableUploads(data.uploads);

        // If we want to verify saved path existence or auto-select logic
        const savedPath = localStorage.getItem('last_uploaded_path');
        if (savedPath && data.uploads.find((u: any) => u.path === savedPath)) {
          fetchUploadedFiles(savedPath);
        }
      })
      .catch(e => console.error("Failed to fetch uploads history", e));
  };

  // Initial Fetch
  React.useEffect(() => {
    fetchUploads();
  }, []);

  const handleSetUploadedPath = (path: string) => {
    setUploadedPath(path);
    localStorage.setItem('last_uploaded_path', path);
    fetchUploadedFiles(path); // Refresh file list for this path
  };
  const [subjectIdInput, setSubjectIdInput] = useState("sub-01");
  const [conversionStatus, setConversionStatus] = useState<'idle' | 'converting' | 'done' | 'error'>('idle');
  /* Subject Selection Handler */
  const [conversionMsg, setConversionMsg] = useState("");
  const handleFileClick = (file: string) => {
    // Only allow selection of directories starting with 'sub-'
    // In our list logic, directories end with '/'
    if (file.endsWith('/') && file.startsWith('sub-')) {
      const subject = file.replace(/\/$/, ''); // Remove trailing slash
      setSelectedSubject(subject);

      // Trigger Validation (Mock for now, or call backend if available)
      // Since it's from Converted Data, we assume valid BIDS structure
      setValidationReport({
        valid: true,
        issues: [],
        pipeline_config: {
          motion_correction: true,
          slice_timing: true,
          coregistration: true,
          segmentation: true,
          normalization: true,
          smoothing: true
        }
      });

      // Reset pipeline status if switching subjects
      if (selectedSubject !== subject) {
        setPipelineStatus('idle');
        setPipelineMsg("");
      }
    }
  };

  // Pipeline State
  const [pipelineStatus, setPipelineStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');
  const [pipelineMsg, setPipelineMsg] = useState("");
  const [pipelineLogs, setPipelineLogs] = useState<string[]>([]);

  // Poll Pipeline Status
  React.useEffect(() => {
    let interval: NodeJS.Timeout;
    if (pipelineStatus === 'running' && selectedSubject) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`http://localhost:8000/pipeline/status/${selectedSubject}`);
          const data = await res.json();
          if (data.status !== 'unknown') {
            setPipelineStatus(data.status);
            setPipelineMsg(`${data.stage}: ${data.message}`);
            if (data.logs) setPipelineLogs(data.logs);
            if (data.status === 'completed') {
              clearInterval(interval);
            }
          }
        } catch (e) { console.error("Pipeline poll error", e); }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [pipelineStatus, selectedSubject]);

  const startPipeline = async () => {
    if (!selectedSubject) return;
    setPipelineStatus('running');
    setPipelineMsg("Initializing Pipeline...");

    // Derive BIDS dir from upload or convert path
    // For now, assuming standard flow: uploadedPath is the BIDS root
    const bidsDir = uploadedPath;

    try {
      const res = await fetch('http://localhost:8000/pipeline/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subject: selectedSubject,
          bids_dir: bidsDir
        })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to start");
      setPipelineMsg("Pipeline Started...");
    } catch (e: any) {
      setPipelineStatus('failed');
      setPipelineMsg("Error: " + e.message);
    }
  };

  const fetchUploadedFiles = async (path: string) => {
    try {
      const res = await fetch('http://localhost:8000/datasets/list-files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path }),
      });
      const data = await res.json();
      if (res.ok) {
        setUploadedFiles(data.files);
        setUploadedPath(path); // Update internal state
        // We do strictly set localStorage via the handle helper usually, but here ensure sync
        localStorage.setItem('last_uploaded_path', path);
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Polling effect
  React.useEffect(() => {
    let interval: NodeJS.Timeout;

    if (conversionStatus === 'converting' && subjectIdInput) {
      interval = setInterval(async () => {
        try {
          // Handle default sub- prefix logic if needed
          const res = await fetch(`http://localhost:8000/convert/status/${subjectIdInput}`);
          const data = await res.json();
          if (data.status !== 'unknown' && data.status !== 'failed') {
            setConversionMsg(`[${data.percent}%] ${data.stage}`);
            setConversionStatus('done');
            setConversionMsg(`Completed! Saved to: ${data.output_path || 'converted_data'}`);

            // Refresh available datasets to show the new converted_data if not already there
            fetchUploads();

            // OPTIONAL: Auto-switch view to 'Converted Data' if that's where it went
            // We know "Converted Data (BIDS Root)" should be in the list now/soon.
            // Let's rely on user selecting it or simple refresh.
          } else if (data.status === 'failed') {
            setConversionStatus('error');
            setConversionMsg(`Failed: ${data.error}`);
          }
        } catch (e) {
          // ignore network poll errors
        }
      }, 1000);
    }

    return () => clearInterval(interval);
  }, [conversionStatus, subjectIdInput]);

  const startConversion = async () => {
    if (!uploadedPath) return;
    setConversionStatus('converting');
    setConversionMsg("Starting pipeline...");
    try {
      const res = await fetch('http://localhost:8000/convert/pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          input_dir: uploadedPath,
          subject: subjectIdInput
        })
      });
      const data = await res.json();
      if (res.ok) {
        setConversionMsg(data.message + " Waiting for updates...");
        // Status remains 'converting', polling effect will pick it up
      } else {
        setConversionStatus('error');
        setConversionMsg("Failed: " + data.detail);
      }
    } catch (e) {
      setConversionStatus('error');
      setConversionMsg("Network error");
    }
  };

  // Import State
  const [showImport, setShowImport] = useState(false);
  const [importConfig, setImportConfig] = useState({
    dataType: 'bids', // 'bids' or 'dicom'
    subject_id: '',
    t1_path: '',
    bold_path: '',
    tr: 2.0
  });

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 p-8 font-sans relative">



      {/* Import Modal */}
      {showImport && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-950 p-8 rounded-2xl border border-slate-700 w-full max-w-2xl shadow-2xl relative overflow-hidden">
            {/* Gradient Border Effect */}
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-blue-600 to-cyan-500"></div>

            <div className="mb-6">
              <div className="flex bg-slate-900 rounded-lg p-1 border border-slate-700 w-fit mb-4">
                <button
                  onClick={() => setImportConfig({ ...importConfig, dataType: 'bids' })}
                  className={cn("px-4 py-2 rounded-md text-sm font-medium transition-all", importConfig.dataType === 'bids' ? "bg-slate-700 text-white shadow-sm" : "text-slate-400 hover:text-white")}
                >
                  Existing BIDS Dataset
                </button>
                <button
                  onClick={() => setImportConfig({ ...importConfig, dataType: 'dicom' })}
                  className={cn("px-4 py-2 rounded-md text-sm font-medium transition-all", importConfig.dataType === 'dicom' ? "bg-slate-700 text-white shadow-sm" : "text-slate-400 hover:text-white")}
                >
                  Raw DICOM
                </button>
              </div>

              {importConfig.dataType === 'dicom' && (
                <div className="bg-slate-900/50 p-4 rounded-lg border border-slate-800 mb-4">
                  <label className="block text-sm text-slate-400 mb-2">Subject ID (Required)</label>
                  <input
                    type="text"
                    className="bg-slate-950 border border-slate-700 rounded px-3 py-2 w-full text-white focus:ring-2 focus:ring-cyan-500 outline-none"
                    placeholder="e.g. sub-01"
                    value={importConfig.subject_id}
                    onChange={(e) => setImportConfig({ ...importConfig, subject_id: e.target.value })}
                  />
                  <p className="text-xs text-amber-400 mt-2 flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" />
                    Files will be converted to NIfTI automatically.
                  </p>
                </div>
              )}
            </div>

            <Dropzone
              onCancel={() => setShowImport(false)}
              onUploadComplete={async (path) => {
                if (importConfig.dataType === 'dicom') {
                  await fetchUploadedFiles(path);
                } else {
                  // alert(`Files uploaded to: ${path}`);
                  await fetchUploadedFiles(path); // This will update state and localStorage
                }
                setShowImport(false);
              }}
            />
          </div>
        </div>
      )}


      <header className="mb-10 flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 to-blue-600 bg-clip-text text-transparent">
            fMRI Preproc
          </h1>
          <p className="text-slate-400 mt-2">Advanced Neuroimaging Pipeline Interface</p>
        </div>
        <div className="flex gap-4">
          <button
            onClick={() => setShowImport(true)}
            className="bg-slate-800 hover:bg-slate-700 px-4 py-2 rounded-lg border border-slate-700 text-cyan-400 font-medium transition-colors flex items-center gap-2"
          >
            + Import New Data
          </button>
          <div className="bg-slate-900 px-4 py-2 rounded-lg border border-slate-800 flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
            <span className="text-sm font-medium">System Ready</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto">
        {/* File Enumeration Panel */}
        {/* Show if we have a path or files (even if empty list, showing container is good if path indicates valid dir) */}
        {(uploadedPath || uploadedFiles.length > 0) && (
          <section className="mb-8 animate-in fade-in slide-in-from-top-4 duration-500">
            <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold flex items-center gap-2 text-white">
                  <Activity className="w-5 h-5 text-cyan-400" />
                  Uploaded Files
                </h2>
                <div className="flex items-center gap-2">
                  <select
                    value={uploadedPath}
                    onChange={(e) => handleSetUploadedPath(e.target.value)}
                    className="bg-slate-950 border border-slate-700 text-xs rounded px-2 py-1 text-slate-400 max-w-[200px]"
                  >
                    <option value="" disabled>Select upload...</option>
                    {availableUploads.map(upl => (
                      <option key={upl.path} value={upl.path}>{upl.name}</option>
                    ))}
                    {uploadedPath && !availableUploads.find(u => u.path === uploadedPath) && (
                      <option value={uploadedPath}>Current</option>
                    )}
                  </select>
                </div>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3 max-h-96 overflow-y-auto custom-scrollbar p-1">
                {uploadedFiles.map((file, idx) => {
                  const isSubject = file.endsWith('/') && file.startsWith('sub-');
                  const isSelected = selectedSubject && file.startsWith(selectedSubject);
                  return (
                    <div
                      key={idx}
                      onClick={() => handleFileClick(file)}
                      className={cn(
                        "bg-slate-950/50 p-3 rounded-lg border border-slate-800/50 flex items-center gap-3 transition-all",
                        isSubject ? "cursor-pointer hover:border-cyan-500/50 hover:bg-slate-900" : "opacity-60",
                        isSelected ? "border-cyan-500 bg-slate-900 ring-1 ring-cyan-500/50" : ""
                      )}
                    >
                      <div className={cn("w-2 h-2 rounded-full", isSubject ? "bg-cyan-500" : "bg-slate-600")} />
                      <span className="text-sm font-mono text-slate-300 truncate" title={file}>
                        {file}
                      </span>
                    </div>
                  );
                })}
              </div>
              <div className="mt-4 pt-4 border-t border-slate-800 flex items-center gap-4">
                <div className="flex-1">
                  <label className="text-xs text-slate-500 block mb-1">Subject ID for Conversion</label>
                  <input
                    type="text"
                    value={subjectIdInput}
                    onChange={(e) => setSubjectIdInput(e.target.value)}
                    className="bg-slate-950 border border-slate-700 rounded px-3 py-2 w-full text-sm text-white"
                  />
                </div>
                <button
                  onClick={startConversion}
                  disabled={conversionStatus === 'converting'}
                  className="bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded-lg font-medium text-sm flex items-center gap-2 mt-4"
                >
                  {conversionStatus === 'converting' ? <Activity className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  Convert to NIfTI
                </button>
              </div>
              {conversionMsg && (
                <div className={cn("mt-2 text-xs p-2 rounded", conversionStatus === 'error' ? "bg-red-900/20 text-red-400" : "bg-green-900/20 text-green-400")}>
                  {conversionMsg}
                </div>
              )}
            </div>
          </section >
        )
        }


        {/* Right Column: details & pipeline */}
        <section className="space-y-6">
          {selectedSubject && validationReport ? (
            <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold text-white flex items-center gap-3">
                    {selectedSubject}
                    {validationReport.valid ? (
                      <span className="px-3 py-1 bg-green-500/10 text-green-400 text-xs rounded-full border border-green-500/20">Ready to Process</span>
                    ) : (
                      <span className="px-3 py-1 bg-red-500/10 text-red-400 text-xs rounded-full border border-red-500/20">Issues Found</span>
                    )}
                  </h2>

                  <div className="flex gap-2">
                    <button
                      onClick={startPipeline}
                      disabled={pipelineStatus === 'running'}
                      className={cn(
                        "text-white px-6 py-2 rounded-lg font-semibold shadow-lg transition-all flex items-center gap-2",
                        pipelineStatus === 'running'
                          ? "bg-slate-700 cursor-not-allowed"
                          : "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 shadow-cyan-900/20"
                      )}
                    >
                      {pipelineStatus === 'running' ? (
                        <><Activity className="w-4 h-4 animate-spin" /> Processing...</>
                      ) : (
                        <><Play className="w-4 h-4" /> Run Standard Pipeline</>
                      )}
                    </button>
                  </div>
                  {pipelineMsg && (
                    <div className="mt-2 text-right">
                      <span className={cn(
                        "text-xs px-2 py-1 rounded border",
                        pipelineStatus === 'failed' ? "bg-red-900/20 border-red-800 text-red-300" :
                          pipelineStatus === 'completed' ? "bg-green-900/20 border-green-800 text-green-300" :
                            "bg-slate-800 border-slate-700 text-cyan-400"
                      )}>
                        {pipelineMsg}
                      </span>
                    </div>
                  )}
                </div>

                {/* Live Logs / Step History */}
                {pipelineStatus !== 'idle' && (
                  <div className="bg-slate-950/50 rounded-lg p-4 border border-slate-800 max-h-60 overflow-y-auto font-mono text-xs space-y-1">
                    <h4 className="text-slate-500 uppercase tracking-wider mb-2 text-[10px]">Processing Log</h4>
                    {pipelineLogs.map((log, i) => (
                      <div key={i} className="text-slate-300 border-l-2 border-slate-800 pl-2 py-0.5">
                        {log}
                        {log.includes("Completed") && <span className="ml-2 text-green-500">✓</span>}
                      </div>
                    ))}
                    {pipelineStatus === 'running' && (
                      <div className="text-cyan-500 animate-pulse pl-2">_</div>
                    )}
                  </div>
                )}

                <div className="space-y-4">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Validation Report</h3>
                  {validationReport.issues.length === 0 ? (
                    <div className="p-4 bg-green-950/30 border border-green-900/50 rounded-lg flex items-center gap-3">
                      <CheckCircle className="w-5 h-5 text-green-400" />
                      <span className="text-green-200">All checks passed. Metadata is complete.</span>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      {validationReport.issues.map((issue: ValidationIssue, idx: number) => (
                        <div key={idx} className={cn(
                          "p-4 border rounded-lg flex items-start gap-3",
                          issue.severity === 'critical' ? "bg-red-950/30 border-red-900/50" : "bg-yellow-950/30 border-yellow-900/50"
                        )}>
                          <AlertCircle className={cn("w-5 h-5 mt-0.5", issue.severity === 'critical' ? "text-red-400" : "text-yellow-400")} />
                          <div>
                            <p className={cn("font-medium", issue.severity === 'critical' ? "text-red-300" : "text-yellow-300")}>
                              {issue.message}
                            </p>
                            <p className="text-xs text-slate-500 mt-1 font-mono">{issue.scan_path}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <div className="mt-8 pt-6 border-t border-slate-800">
                  <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Pipeline Configuration</h3>
                  <div className="grid grid-cols-2 gap-4">
                    {validationReport.pipeline_config && Object.entries(validationReport.pipeline_config).map(([key, value]) => (
                      <div key={key} className="flex items-center justify-between p-3 bg-slate-950 rounded-lg border border-slate-800">
                        <span className="text-slate-300 capitalize">{key.replace(/_/g, ' ')}</span>
                        <div className={cn("w-2 h-2 rounded-full", value ? "bg-cyan-400 box-shadow-cyan" : "bg-slate-600")} />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-slate-600 border-2 border-dashed border-slate-800 rounded-3xl min-h-[400px]">
              <Activity className="w-16 h-16 mb-4 opacity-20" />
              <p className="text-lg">Select a subject to view details</p>
            </div>
          )}
        </section>
      </main >
    </div >
  );
}
