import type { Metadata } from "next";
import localFont from "next/font/local";
import Link from "next/link";
import "./globals.css";

const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "E-CIP Dashboard",
  description: "E-Commerce Intelligence Platform — operations dashboard",
};

const NAV_ITEMS = [
  { href: "/", label: "Overview" },
  { href: "/products", label: "Product Analytics" },
  { href: "/sentiment", label: "Sentiment Analytics" },
  { href: "/retention", label: "Retention Analytics" },
  { href: "/review-queue", label: "Review Queue" },
  { href: "/drift", label: "Drift Monitor" },
];

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased bg-neutral-950 text-neutral-100`}>
        <nav className="border-b border-neutral-800 bg-neutral-900/80 backdrop-blur sticky top-0 z-10">
          <div className="max-w-6xl mx-auto px-6 h-14 flex items-center gap-6">
            <span className="font-semibold tracking-tight text-teal-400">
              E-CIP <span className="text-neutral-500 text-xs">v3.0</span>
            </span>
            <div className="flex gap-4 text-sm">
              {NAV_ITEMS.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="text-neutral-400 hover:text-neutral-100 transition-colors"
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </div>
        </nav>
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
