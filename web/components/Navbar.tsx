"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';

export default function Navbar() {
    const pathname = usePathname();

    // Reverted to minimal navigation as per user request
    const navItems = [
        { name: 'Dashboard', href: '/' },
        { name: 'Realign', href: '/realign' },
        { name: 'Slice Timing', href: '/stc' },
    ];

    return (
        <nav className="w-full bg-white border-b border-gray-200 sticky top-0 z-40">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-16">
                    {/* Navigation Links */}
                    <div className="flex items-center space-x-1 overflow-x-auto no-scrollbar">
                        {navItems.map((item) => {
                            const isActive = pathname === item.href;
                            return (
                                <Link
                                    key={item.name}
                                    href={item.href}
                                    className={`px-4 py-2 rounded-lg text-sm font-bold uppercase tracking-wider transition-all duration-200 whitespace-nowrap no-underline
                    ${isActive
                                            ? 'text-emerald-700 bg-emerald-50 border border-emerald-200 font-extrabold'
                                            : 'text-gray-600 hover:text-black hover:bg-gray-50'
                                        }`}
                                >
                                    {item.name}
                                </Link>
                            );
                        })}
                    </div>
                </div>
            </div>
        </nav>
    );
}
