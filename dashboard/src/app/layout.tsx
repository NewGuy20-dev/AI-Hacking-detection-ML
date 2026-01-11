import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { Providers } from '@/components/Providers'
import { Header, Sidebar } from '@/components/layout'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'AI Hacking Detection',
  description: 'Real-time cyber attack detection using ensemble ML models',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <a href="#main-content" className="skip-link">
            Skip to main content
          </a>
          <div className="min-h-screen">
            <div className="max-w-7xl mx-auto px-4 py-6">
              <Header />
              <div className="flex gap-6">
                <Sidebar />
                <main id="main-content" className="flex-1 min-w-0">
                  {children}
                </main>
              </div>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  )
}
