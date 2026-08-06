interface AlertFeature {
  feature: string
  shap_value: number
}

interface AlertEvent {
  type: string
  id?: number
  probability: number
  threshold: number
  top_features: AlertFeature[]
  transaction: {
    Amount: number
    Time: number
    [key: string]: number
  }
}

interface FeedItem {
  id: number
  amount: number
  isFraud: boolean
  probability: number
}

export type { AlertEvent, AlertFeature, FeedItem }
