import { useEffect } from 'react'
import type { Transaction, ShapFeature } from './store'
import {
  IconXmark,
  IconXmarkCircle,
  IconCheckCircle,
  IconDollar,
  IconClock,
  IconLayers,
  IconChart,
} from './icons'

interface Props {
  tx: Transaction | null
  onClose: () => void
}

function parseFeatures(json: string): ShapFeature[] {
  try {
    return JSON.parse(json)
  } catch {
    return []
  }
}

function fmtTime(seconds: number): string {
  const h = Math.floor(seconds / 3600) % 24
  const m = Math.floor(seconds / 60) % 60
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`
}

function fmtDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

export function TxDetailModal({ tx, onClose }: Props) {
  useEffect(() => {
    if (!tx) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [tx, onClose])

  if (!tx) return null

  const features = parseFeatures(tx.top_features)
  const isFraud = tx.is_fraud === 1
  const maxVal = features.length > 0
    ? Math.max(...features.map((f) => Math.abs(f.shap_value)))
    : 1
  const vFields = Array.from({ length: 28 }, (_, i) => `v${i + 1}`) as (keyof Transaction)[]

  return (
    <dialog
      className="modal modal-open"
      onClick={onClose}
    >
      <div
        className="modal-box max-w-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <span
                className={`badge ${isFraud ? 'badge-error' : 'badge-success'} gap-1`}
              >
                {isFraud ? (
                  <>
                    <IconXmarkCircle size={14} />
                    Fraud Detected
                  </>
                ) : (
                  <>
                    <IconCheckCircle size={14} />
                    Legitimate
                  </>
                )}
              </span>
              <span className="text-3xl font-bold flex items-center gap-1">
                <IconDollar size={24} className="text-base-content/40" />
                {tx.amount.toFixed(2)}
              </span>
            </div>
            <div className="text-sm text-base-content/50 flex items-center gap-1.5">
              <IconClock size={12} className="text-base-content/30" />
              ID #{tx.id} · {fmtDate(tx.timestamp)}
            </div>
          </div>
          <button
            className="btn btn-sm btn-circle btn-ghost"
            onClick={onClose}
          >
            <IconXmark size={18} />
          </button>
        </div>

        <div className="grid grid-cols-3 gap-3 mt-5">
          <div className="stat bg-base-300 rounded-box py-3 px-4">
            <div className="stat-title text-xs">Probability</div>
            <div
              className={`stat-value text-2xl ${
                isFraud ? 'text-error' : 'text-success'
              }`}
            >
              {(tx.probability * 100).toFixed(1)}%
            </div>
          </div>
          <div className="stat bg-base-300 rounded-box py-3 px-4">
            <div className="stat-title text-xs">Threshold</div>
            <div className="stat-value text-2xl">
              {(tx.threshold * 100).toFixed(1)}%
            </div>
          </div>
          <div className="stat bg-base-300 rounded-box py-3 px-4">
            <div className="stat-title text-xs">Time</div>
            <div className="stat-value text-2xl font-mono">
              {fmtTime(tx.time)}
            </div>
          </div>
        </div>

        {features.length > 0 && (
          <div className="mt-5 space-y-3">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
              <IconChart size={16} className="text-primary" />
              Feature Contributions
            </h3>
            <div className="flex flex-col gap-2">
              {features.map((f, i) => {
                const widthPct = (Math.abs(f.shap_value) / maxVal) * 100
                const positive = f.shap_value > 0
                return (
                  <div key={i} className="flex items-center gap-3">
                    <span className="text-xs font-mono w-16 shrink-0">
                      {f.feature}
                    </span>
                    <div className="flex-1 h-3 bg-base-300 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          positive ? 'bg-error/60' : 'bg-info/60'
                        }`}
                        style={{ width: `${widthPct}%` }}
                      />
                    </div>
                    <span
                      className={`text-xs font-mono w-16 text-right shrink-0 ${
                        positive ? 'text-error' : 'text-info'
                      }`}
                    >
                      {positive ? '+' : ''}
                      {f.shap_value.toFixed(4)}
                    </span>
                  </div>
                )
              })}
            </div>
            <div className="flex gap-4 text-xs text-base-content/40">
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-error" />
                Pushes toward fraud
              </span>
              <span className="flex items-center gap-1">
                <span className="w-2.5 h-2.5 rounded-full bg-info" />
                Pushes toward legitimate
              </span>
            </div>
          </div>
        )}

        <div className="mt-5 space-y-3">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
            <IconLayers size={16} className="text-primary" />
            PCA Features
          </h3>
          <div className="grid grid-cols-4 gap-1.5">
            {vFields.map((v) => (
              <div
                key={v}
                className="bg-base-300/50 rounded px-2 py-1.5 border border-base-300"
              >
                <div className="text-[10px] text-base-content/50 font-semibold uppercase">
                  {v}
                </div>
                <div className="text-xs font-mono">
                  {tx[v] !== undefined ? (tx[v] as number).toFixed(4) : '—'}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="modal-action">
          <button className="btn btn-sm" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </dialog>
  )
}
