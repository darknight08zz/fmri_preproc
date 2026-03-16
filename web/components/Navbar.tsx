"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { name: "Dashboard",      href: "/",         step: null },
  { name: "Realign",        href: "/realign",  step: "01" },
  { name: "Slice Timing",   href: "/stc",      step: "02" },
  { name: "Coregistration", href: "/coreg",    step: "03" },
  { name: "Segmentation",   href: "/segment",  step: "04" },
  { name: "Normalisation",  href: "/normalise",step: "05" },
  { name: "Smoothing",      href: "/smooth",   step: "06" },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <nav className="w-full bg-black/20 backdrop-blur-md border-b border-white/[0.04] sticky top-0 z-40 px-6">
      <div className="max-w-[1400px] mx-auto h-12 flex items-center gap-2 overflow-x-auto">

        {/* Nav links */}
        {navItems.map((item) => {
          const isActive = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`
                flex items-center gap-1.5 px-[10px] py-1.5 rounded-md
                text-[13px] font-mono whitespace-nowrap no-underline flex-shrink-0
                border transition-all duration-150
                ${isActive
                  ? "text-[#34d399] bg-[#10b98114] border-[#10b98140]"
                  : "text-white/30 border-transparent hover:text-white/60 hover:bg-white/[0.03]"
                }
              `}
            >
              {item.step && (
                <span className={`text-[11px] ${isActive ? "text-[#34d39980]" : "text-white/15"}`}>
                  {item.step}
                </span>
              )}
              {item.name}
            </Link>
          );
        })}

        {/* Progress dots — right side */}
        <div className="ml-auto flex-shrink-0 flex items-center gap-[5px] pl-[14px] border-l border-white/[0.05]">
          {navItems.slice(1).map((item) => {
            const activeIdx   = navItems.findIndex((n) => n.href === pathname);
            const thisIdx     = navItems.indexOf(item);
            const isActive    = pathname === item.href;
            const isDone      = activeIdx > thisIdx;
            return (
              <div
                key={item.href}
                className={`
                  w-[5px] h-[5px] rounded-full transition-all duration-300
                  ${isActive ? "bg-[#34d399] scale-[1.3]"
                  : isDone  ? "bg-[#064e3b]"
                  :           "bg-white/[0.08]"
                }`}
              />
            );
          })}
        </div>

      </div>
    </nav>
  );
}