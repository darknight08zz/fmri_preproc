"use client";

import React, { useState } from 'react';
import { FolderSearch, Play, AlertCircle, CheckCircle, Brain, Activity, Settings } from 'lucide-react';
import Dropzone from './Dropzone';
import PipelineConverter from './PipelineConverter';
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
  const [bidsPath, setBidsPath] = useState('');

  // Load from localStorage on mount
  React.useEffect(() => {
    const savedPath = localStorage.getItem('fmri_bids_path');
    if (savedPath) {
      setBidsPath(savedPath);
    }
  }, []);

  // Save to localStorage whenever it changes
  React.useEffect(() => {
    if (bidsPath) {
      localStorage.setItem('fmri_bids_path', bidsPath);
    }
  }, [bidsPath]);
  const [subjects, setSubjects] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [availableDatasets, setAvailableDatasets] = useState<{ name: string, path: string }[]>([]); // New state
  const [selectedSubject, setSelectedSubject] = useState<string | null>(null);
  const [validationReport, setValidationReport] = useState<ValidationReport | null>(null);

  // New State for Pipeline Converter
  const [showPipeline, setShowPipeline] = useState(false);

  // Fetch available datasets on mount
  React.useEffect(() => {
    fetch('http://localhost:8000/datasets/available')
      .then(res => res.json())
      .then(data => {
        setAvailableDatasets(data.datasets);
        // If current bidsPath is empty and we have datasets, default to first
        setBidsPath(prev => {
          if (!prev && data.datasets.length > 0) return data.datasets[0].path;
          return prev;
        });
      })
      .catch(err => console.error("Failed to fetch datasets", err));
  }, []);

  const fetchSubjects = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/datasets/subjects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: bidsPath }),
      });
      const data = await res.json();
      if (res.ok) {
        setSubjects(data.subjects);
      } else {
        alert("Error fetching subjects: " + data.detail);
      }
    } catch (e) {
      console.error(e);
      alert("Failed to connect to API");
    } finally {
      setLoading(false);
    }
  };

  const validateSubject = async (sub: string) => {
    setSelectedSubject(sub);
    setValidationReport(null); // clear previous
    try {
      const res = await fetch('http://localhost:8000/validation/subject', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ bids_path: bidsPath, subject: sub }),
      });
      const data = await res.json();
      if (res.ok) {
        setValidationReport(data);
      }
    } catch (e) {
      console.error(e);
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

      {/* Pipeline Converter Modal */}
      {showPipeline && selectedSubject && (
        <PipelineConverter
          inputDir={bidsPath} // Using the current path as input source for now, assuming it points to raw DICOMs or BIDS
          subject={selectedSubject}
          onClose={() => setShowPipeline(false)}
        />
      )}

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
                  if (!importConfig.subject_id) {
                    alert("Subject ID is required for DICOM conversion.");
                    return;
                  }

                  const confirmConv = confirm(`Files uploaded to ${path}. Start DICOM conversion for ${importConfig.subject_id}?`);
                  if (confirmConv) {
                    try {
                      // Trigger basic conversion (dcm2niix) for quick import, or allow Pipeline?
                      // For "Import", we probably want dcm2niix.
                      // For "Tool Builder" demo, we use the button in the main UI.
                      const res = await fetch('http://localhost:8000/convert/dicom', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                          input_dir: path,
                          subject: importConfig.subject_id
                        }),
                      });
                      const data = await res.json();
                      if (res.ok) {
                        alert(`Conversion Started! Output will be in: ${data.target_output_dir}`);
                        // Update dashboard to point to the conversion output directory
                        setBidsPath("converted_data");
                        setImportConfig(prev => ({ ...prev, subject_id: '' })); // Reset
                      } else {
                        alert("Conversion failed to start: " + data.detail);
                      }
                    } catch (e) {
                      alert("Failed to connect to Conversion API");
                    }
                  }
                } else {
                  alert(`Files uploaded to: ${path}`);
                  setBidsPath(path);
                }
                setShowImport(false);
                setTimeout(fetchSubjects, 500);
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

      <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Data Selection */}
        <section className="space-y-6">
          {/* Data Source Panel */}
          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <FolderSearch className="w-5 h-5 text-cyan-400" />
              Data Source
            </h2>
            <div className="flex gap-2 mb-2">
              <div className="relative flex-1">
                <select
                  value={bidsPath}
                  onChange={(e) => setBidsPath(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-700 rounded-lg px-4 py-2 appearance-none focus:ring-2 focus:ring-cyan-500 outline-none transition-all text-white"
                >
                  <option value="" disabled>Select a dataset...</option>
                  {availableDatasets.map(ds => (
                    <option key={ds.path} value={ds.path}>
                      📂 {ds.name}
                    </option>
                  ))}
                  {!availableDatasets.find(d => d.path === bidsPath) && bidsPath && (
                    <option value={bidsPath}>{bidsPath} (Custom)</option>
                  )}
                </select>
                <div className="absolute right-3 top-1/2 -translate-y-1/2 pointer-events-none text-slate-500">
                  ▼
                </div>
              </div>
              <button
                onClick={fetchSubjects}
                disabled={loading}
                className="bg-cyan-600 hover:bg-cyan-500 text-white px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
              >
                {loading ? 'Submitting...' : 'Load'}
              </button>
            </div>
            <p className="text-xs text-slate-500">
              Select a dataset from the list. Uploaded/Converted data will appear here automatically.
            </p>
          </div>

          <div className="bg-slate-900 p-6 rounded-2xl border border-slate-800 shadow-xl h-[600px] overflow-hidden flex flex-col">
            <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <Brain className="w-5 h-5 text-purple-400" />
              Subjects
            </h2>
            <div className="flex-1 overflow-y-auto space-y-2 pr-2 custom-scrollbar">
              {subjects.length === 0 && (
                <p className="text-slate-500 text-center mt-10">No subjects loaded.</p>
              )}
              {subjects.map(sub => (
                <div
                  key={sub}
                  onClick={() => validateSubject(sub)}
                  className={cn(
                    "p-3 rounded-xl border border-slate-800 cursor-pointer transition-all hover:bg-slate-800",
                    selectedSubject === sub ? "bg-slate-800 border-cyan-500/50 ring-1 ring-cyan-500/20" : "bg-slate-950/50"
                  )}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-sm">{sub}</span>
                    <Activity className="w-4 h-4 text-slate-600" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Right Column: details & pipeline */}
        <section className="lg:col-span-2 space-y-6">
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
                      onClick={() => setShowPipeline(true)}
                      className="bg-slate-800 hover:bg-slate-700 text-cyan-400 border border-slate-700 px-4 py-2 rounded-lg font-semibold shadow-lg transition-all flex items-center gap-2"
                    >
                      <Settings className="w-4 h-4" /> Pipeline Details
                    </button>
                    <button
                      className="bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 text-white px-6 py-2 rounded-lg font-semibold shadow-lg shadow-cyan-900/20 transition-all flex items-center gap-2"
                    >
                      <Play className="w-4 h-4" /> Run Standard
                    </button>
                  </div>
                </div>

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
      </main>
    </div>
  );
}
