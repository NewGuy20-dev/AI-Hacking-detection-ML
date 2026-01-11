'use client'

import { ReactNode, useState } from 'react'
import { Sidebar } from '@/components/layout/Sidebar'
import { Header } from '@/components/layout/Header'
import { ThemeProvider } from 'next-themes'
import { Toaster } from 'sonner'
import { cn } from '@/lib/utils'
import { ErrorBoundary } from '@/components/ui/ErrorBoundary'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './globals.css'

export default function RootLayout({ children }: { children: ReactNode }) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [queryClient] = useState(() => new QueryClient({
    defaultOptions: {
      queries: { staleTime: 60 * 1000, retry: 1 },
    },
  }))

  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
            <div className="min-h-screen">
              <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} />
              <Header sidebarCollapsed={sidebarCollapsed} />
              <main 
                id="main-content"
                className={cn(
                  'p-4 md:p-8 transition-all duration-300',
                  sidebarCollapsed ? 'md:ml-16' : 'md:ml-64'
                )}
                role="main"
                tabIndex={-1}
              >
                <div className="max-w-7xl mx-auto">
                  <ErrorBoundary>
                    {children}
                  </ErrorBoundary>
                </div>
              </main>
            </div>
            <Toaster position="top-right" richColors />
          </ThemeProvider>
        </QueryClientProvider>
      </body>
    </html>
  )
}
