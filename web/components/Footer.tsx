export default function Footer() {
  const pipeline = ["DICOM", "NIfTI", "STC", "Realign", "Coreg", "Seg", "Norm", "Smooth"];

  return (
    <footer className="w-full bg-transparent border-t border-white/[0.05] mt-auto px-6">
      <div className="max-w-[1400px] mx-auto py-5 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">

        {/* Brand */}
        <div>
          <div className="text-white/60 font-black text-sm tracking-tighter leading-none mb-1">
            fMRI<span className="text-[#34d399]">Preproc</span>
          </div>
          <div className="text-white/15 text-[11px] font-light tracking-wide font-sans">
            SPM-based rs-fMRI preprocessing · ADNI-compatible
          </div>
        </div>

        {/* Pipeline breadcrumb */}
        <div className="hidden md:flex items-center gap-1 text-[11px] text-white/15 font-mono">
          {pipeline.map((s, i) => (
            <span key={s} className="flex items-center gap-1">
              <span>{s}</span>
              {i < pipeline.length - 1 && (
                <span className="text-[#064e3b]">→</span>
              )}
            </span>
          ))}
        </div>

        {/* Links + year */}
        <div className="flex items-center gap-4 text-[12px] text-white/20 font-mono">
          <a href="#" className="hover:text-[#34d399] transition-colors">Docs</a>
          <a href="#" className="hover:text-[#34d399] transition-colors">Support</a>
          <a href="#" className="hover:text-[#34d399] transition-colors">GitHub</a>
          <span className="text-white/5 opacity-40">© 2026</span>
        </div>

      </div>
    </footer>
  );
}