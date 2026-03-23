import type { Metadata } from 'next'
import { Chakra_Petch, Fira_Code } from 'next/font/google'
import './globals.css'
import { Providers } from '@/components/Providers'
import { Header, Sidebar } from '@/components/layout'

const chakra = Chakra_Petch({ 
  weight: ['300', '400', '500', '600', '700'],
  subsets: ['latin'],
  variable: '--font-chakra',
})

const firaCode = Fira_Code({
  subsets: ['latin'],
  variable: '--font-fira-code',
})

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
    <html lang="en" suppressHydrationWarning className="dark">
      <body className={`${chakra.variable} ${firaCode.variable} font-sans antialiased text-cyber-text bg-cyber-bg min-h-screen selection:bg-primary/30 selection:text-white`}>
        <Providers>
          <a href="#main-content" className="skip-link">
            [ SKIP TO MAIN CONTENT ]
          </a>
          <div className="relative z-10 min-h-screen flex flex-col">
            <div className="w-full border-b border-cyber-border/50 bg-cyber-bg/80 backdrop-blur-md sticky top-0 z-50">
              <div className="max-w-[1600px] mx-auto px-4 sm:px-6 lg:px-8">
                <Header />
              </div>
            </div>
            <div className="max-w-[1600px] mx-auto w-full px-4 sm:px-6 lg:px-8 py-6 flex-1 flex flex-col md:flex-row gap-8">
              <Sidebar />
              <main id="main-content" className="flex-1 min-w-0 relative pb-12">
                {children}
              </main>
            </div>
          </div>
        </Providers>
      </body>
    </html>
  )
}
