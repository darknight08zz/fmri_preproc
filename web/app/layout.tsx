import type { Metadata } from "next";
import { JetBrains_Mono } from "next/font/google";
import "./globals.css";
import Header from "@/components/Header";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["300", "400", "500", "700", "800"],
});

export const metadata: Metadata = {
  title: "fMRI Preproc — SPM12 Pipeline",
  description: "Advanced rs-fMRI preprocessing pipeline · ADNI-compatible · SPM12",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body
        className={`
          ${jetbrainsMono.variable}
          font-mono
          min-h-screen flex flex-col
          text-white
          antialiased
        `}
      >
        <Header />
        <Navbar />
        <main className="flex-grow flex flex-col relative z-0">
          {children}
        </main>
        <Footer />
      </body>
    </html>
  );
}