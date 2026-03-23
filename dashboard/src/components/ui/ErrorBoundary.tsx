'use client'

import { Component, ReactNode } from 'react'
import { AlertTriangle, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/Button'

interface Props {
  children: ReactNode
  fallback?: ReactNode
}

interface State {
  hasError: boolean
  error?: Error
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback || (
        <div className="flex flex-col items-center justify-center p-8 bg-black/60 backdrop-blur-md border border-danger/50 relative overflow-hidden group">
          <div className="absolute inset-0 bg-cyber-grid opacity-10 pointer-events-none" />
          <div className="absolute top-0 left-0 w-2 h-full bg-danger shadow-neon-danger" />
          <div className="p-4 bg-black border border-danger/30 mb-4 relative z-10">
            <AlertTriangle className="w-8 h-8 text-danger animate-pulse" />
          </div>
          <h3 className="text-lg font-bold font-mono tracking-widest uppercase text-danger mb-2 relative z-10">
            [ FATAL_SYS_ERROR ]
          </h3>
          <p className="text-sm font-mono tracking-wider text-cyber-muted mb-6 text-center max-w-md uppercase relative z-10">
            {this.state.error?.message || 'An unexpected anomaly occurred'}
          </p>
          <Button 
            onClick={() => this.setState({ hasError: false })}
            variant="danger"
            className="uppercase tracking-widest font-mono text-[10px] relative z-10"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            [ RESET_AND_RETRY ]
          </Button>
        </div>
      )
    }

    return this.props.children
  }
}
