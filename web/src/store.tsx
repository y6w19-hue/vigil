import {
	createContext,
	useCallback,
	useContext,
	useEffect,
	useRef,
	useState,
	type ReactNode,
} from "react";
import Ably from "ably";
import { toast } from "sonner";
import { playAlertSound } from "./sound";

const ABLY_KEY = import.meta.env.VITE_ABLY_KEY as string;
const API_URL = `http://${window.location.hostname}:8000`;

export interface ShapFeature {
	feature: string;
	shap_value: number;
}

export interface Transaction {
	id: number;
	timestamp: string;
	amount: number;
	time: number;
	probability: number;
	is_fraud: number;
	threshold: number;
	top_features: string;
	v1: number;
	v2: number;
	v3: number;
	v4: number;
	v5: number;
	v6: number;
	v7: number;
	v8: number;
	v9: number;
	v10: number;
	v11: number;
	v12: number;
	v13: number;
	v14: number;
	v15: number;
	v16: number;
	v17: number;
	v18: number;
	v19: number;
	v20: number;
	v21: number;
	v22: number;
	v23: number;
	v24: number;
	v25: number;
	v26: number;
	v27: number;
	v28: number;
}

export interface AlertHistoryPoint {
	time: number;
	count: number;
}

interface GlobalState {
	connected: boolean;
	stats: { total: number; fraud: number; alerts: number };
	transactions: Transaction[];
	alerts: Transaction[];
	alertHistory: AlertHistoryPoint[];
	hasMore: boolean;
	loadingMore: boolean;
	loadMore: () => void;
	refresh: () => void;
}

const GlobalContext = createContext<GlobalState | null>(null);

const TX_PAGE_SIZE = 25;

export function GlobalProvider({ children }: { children: ReactNode }) {
	const [connected, setConnected] = useState(false);
	const [stats, setStats] = useState({ total: 0, fraud: 0, alerts: 0 });
	const [transactions, setTransactions] = useState<Transaction[]>([]);
	const [alerts, setAlerts] = useState<Transaction[]>([]);
	const [alertHistory, setAlertHistory] = useState<AlertHistoryPoint[]>([]);
	const [hasMore, setHasMore] = useState(true);
	const [loadingMore, setLoadingMore] = useState(false);
	const offsetRef = useRef(0);
	const ablyRef = useRef<Ably.Realtime | null>(null);

	const fetchInitial = useCallback(async () => {
		try {
			const [statsResp, txResp, alertsResp, historyResp] = await Promise.all([
				fetch(`${API_URL}/stats`).then((r) => r.json()),
				fetch(`${API_URL}/transactions?limit=${TX_PAGE_SIZE}`).then((r) =>
					r.json(),
				),
				fetch(`${API_URL}/alerts?limit=20`).then((r) => r.json()),
				fetch(`${API_URL}/alert-history?buckets=20`).then((r) => r.json()),
			]);
			setStats(statsResp);
			setTransactions(txResp);
			setAlerts(alertsResp);
			setAlertHistory(historyResp);
			offsetRef.current = txResp.length;
			setHasMore(txResp.length === TX_PAGE_SIZE);
		} catch {
		}
	}, []);

	const fetchStatsAndAlerts = useCallback(async () => {
		try {
			const [statsResp, alertsResp, historyResp, txResp] =
				await Promise.all([
					fetch(`${API_URL}/stats`).then((r) => r.json()),
					fetch(`${API_URL}/alerts?limit=20`).then((r) => r.json()),
					fetch(`${API_URL}/alert-history?buckets=20`).then((r) =>
						r.json(),
					),
					fetch(`${API_URL}/transactions?limit=${TX_PAGE_SIZE}`).then(
						(r) => r.json(),
					),
				]);
			setStats(statsResp);
			setAlerts(alertsResp);
			setAlertHistory(historyResp);
			setTransactions((prev) => {
				if (prev.length === 0) {
					offsetRef.current = txResp.length;
					setHasMore(txResp.length === TX_PAGE_SIZE);
					return txResp;
				}
				const existingIds = new Set(prev.map((t) => t.id));
				const newOnes = txResp.filter((t: Transaction) => !existingIds.has(t.id));
				if (newOnes.length === 0) return prev;
				return [...newOnes, ...prev];
			});
		} catch {
		}
	}, []);

	const loadMore = useCallback(async () => {
		if (loadingMore || !hasMore) return;
		setLoadingMore(true);
		try {
			const resp = await fetch(
				`${API_URL}/transactions?limit=${TX_PAGE_SIZE}&offset=${offsetRef.current}`,
			);
			const data: Transaction[] = await resp.json();
			if (data.length > 0) {
				setTransactions((prev) => [...prev, ...data]);
				offsetRef.current += data.length;
			}
			setHasMore(data.length === TX_PAGE_SIZE);
		} catch {
		} finally {
			setLoadingMore(false);
		}
	}, [loadingMore, hasMore]);

	useEffect(() => {
		fetchInitial();
		const interval = setInterval(fetchStatsAndAlerts, 2000);
		return () => clearInterval(interval);
	}, [fetchInitial, fetchStatsAndAlerts]);

	useEffect(() => {
		if (!ABLY_KEY) {
			console.error("VITE_ABLY_KEY not set");
			return;
		}

		const ably = new Ably.Realtime({ key: ABLY_KEY });
		ablyRef.current = ably;

		ably.connection.on("connected", () => setConnected(true));
		ably.connection.on("disconnected", () => setConnected(false));
		ably.connection.on("failed", () => setConnected(false));

		const channel = ably.channels.get("fraud-alerts");
		channel.subscribe("fraud_alert", (msg) => {
			let data: {
				id?: number;
				probability?: number;
				transaction?: { Amount?: number };
				top_features?: { feature: string }[];
			};
			try {
				data = typeof msg.data === "string" ? JSON.parse(msg.data) : msg.data;
			} catch {
				return;
			}

			const id = data.id ?? Date.now();
			const amount = data.transaction?.Amount ?? 0;
			const prob = data.probability ?? 0;
			const topFeature = data.top_features?.[0]?.feature ?? "—";

			playAlertSound();

			toast.error("Fraud Detected", {
				description: `$${amount.toFixed(2)} · ${(prob * 100).toFixed(1)}% probability · Top factor: ${topFeature}`,
				duration: 8000,
				action: {
					label: "View Details",
					onClick: () => {
						window.location.href = `/?alert=${id}`;
					},
				},
			});

			fetchStatsAndAlerts();
		});

		return () => {
			ably.close();
		};
	}, [fetchStatsAndAlerts]);

	return (
		<GlobalContext.Provider
			value={{
				connected,
				stats,
				transactions,
				alerts,
				alertHistory,
				hasMore,
				loadingMore,
				loadMore,
				refresh: fetchInitial,
			}}
		>
			{children}
		</GlobalContext.Provider>
	);
}

export function useGlobal() {
	const ctx = useContext(GlobalContext);
	if (!ctx) throw new Error("useGlobal must be used within GlobalProvider");
	return ctx;
}
