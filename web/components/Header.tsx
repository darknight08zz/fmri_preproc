"use client";

import Link from 'next/link';

export default function Header() {
    return (
        <header className="w-full bg-white border-b border-gray-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-start items-center h-24">
                    <Link href="/" className="group flex items-center gap-4 text-black no-underline">
                        {/* Text - Left Aligned & Prominent & Black */}
                        <h1 className="text-4xl font-black text-black tracking-tighter decoration-0">
                            fMRI Preproc
                        </h1>
                    </Link>
                </div>
            </div>
        </header>
    );
}
