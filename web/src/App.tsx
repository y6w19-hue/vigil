import { useEffect, useState } from "react";
import {
	Area,
	AreaChart,
	ResponsiveContainer,
	Tooltip,
	XAxis,
	YAxis,
} from "recharts";
import { useGlobal, type Transaction as Tx } from "./store";
import { Header } from "./Header";
import { TxDetailModal } from "./TxDetailModal";
import {
	IconChart,
	IconSend,
	IconCreditCard,
	IconBell,
	IconCheckCircle,
	IconXmarkCircle,
	IconDollar,
	IconClock,
	IconEye,
	IconLayers,
} from "./icons";
import { FRAUD_PRESET, LEGIT_PRESET, type TransactionPreset } from "./presets";

const API_URL = `http://${window.location.hostname}:8000`;

function fmtTime(seconds: number): string {
	const h = Math.floor(seconds / 3600) % 24;
	const m = Math.floor(seconds / 60) % 60;
	return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

function App() {
	const {
		stats,
		transactions,
		alerts,
		alertHistory,
		hasMore,
		loadingMore,
		loadMore,
		refresh,
	} = useGlobal();
	const [txForm, setTxForm] = useState<TransactionPreset>(LEGIT_PRESET);
	const [lastResult, setLastResult] = useState<{
		is_fraud: boolean;
		probability: number;
		threshold: number;
		top_features: { feature: string; shap_value: number }[];
	} | null>(null);
	const [submitting, setSubmitting] = useState(false);
	const [selectedTx, setSelectedTx] = useState<Tx | null>(null);

	useEffect(() => {
		const params = new URLSearchParams(window.location.search);
		const alertId = params.get("alert");
		if (!alertId) return;

		fetch(`${API_URL}/transaction/${alertId}`)
			.then((r) => r.json())
			.then((tx) => {
				if (tx && tx.id) setSelectedTx(tx as Tx);
			})
			.catch(() => {});

		const url = new URL(window.location.href);
		url.searchParams.delete("alert");
		window.history.replaceState({}, "", url.toString());
	}, []);

	const legit = stats.total - stats.fraud;

	const chartData = alertHistory.map((p) => ({
		time: new Date(p.time).toLocaleTimeString(),
		count: p.count,
	}));

	const submitTransaction = async (tx: TransactionPreset) => {
		setSubmitting(true);
		try {
			const resp = await fetch(`${API_URL}/predict`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(tx),
			});
			const result = await resp.json();
			setLastResult(result);
			refresh();
		} catch {
		} finally {
			setSubmitting(false);
		}
	};

	const handleFieldChange = (field: keyof TransactionPreset, value: string) => {
		const num = parseFloat(value);
		setTxForm((prev) => ({ ...prev, [field]: isNaN(num) ? 0 : num }));
	};

	const vFields = Array.from(
		{ length: 28 },
		(_, i) => `V${i + 1}`,
	) as (keyof TransactionPreset)[];

	return (
		<div className="min-h-screen flex flex-col">
			<Header />

			<div className="grid grid-cols-1 lg:grid-cols-3 gap-4 p-6 flex-1">
				<div className="card bg-base-200 shadow-sm">
					<div className="card-body space-y-4">
						<h2 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
							<IconChart size={18} className="text-primary" />
							Live Statistics
						</h2>
						<div className="grid grid-cols-2 gap-2">
							<div className="stat bg-base-300 rounded-box">
								<div className="stat-title">Total</div>
								<div className="stat-value text-primary text-2xl">
									{stats.total.toLocaleString()}
								</div>
							</div>
							<div className="stat bg-base-300 rounded-box">
								<div className="stat-title">Legitimate</div>
								<div className="stat-value text-success text-2xl">
									{legit.toLocaleString()}
								</div>
							</div>
							<div className="stat bg-base-300 rounded-box">
								<div className="stat-title">Fraud</div>
								<div className="stat-value text-error text-2xl">
									{stats.fraud.toLocaleString()}
								</div>
							</div>
							<div className="stat bg-base-300 rounded-box">
								<div className="stat-title">Alerts</div>
								<div className="stat-value text-warning text-2xl">
									{stats.alerts.toLocaleString()}
								</div>
							</div>
						</div>

						<div className="space-y-2">
							<h2 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
								<IconBell size={18} className="text-error" />
								Alert Activity
							</h2>
							<div className="h-40">
								<ResponsiveContainer width="100%" height="100%">
									<AreaChart data={chartData}>
										<defs>
											<linearGradient
												id="alertGrad"
												x1="0"
												y1="0"
												x2="0"
												y2="1"
											>
												<stop
													offset="0%"
													stopColor="#ef4444"
													stopOpacity={0.4}
												/>
												<stop
													offset="100%"
													stopColor="#ef4444"
													stopOpacity={0}
												/>
											</linearGradient>
										</defs>
										<XAxis
											dataKey="time"
											tick={{ fill: "#8892a8", fontSize: 10 }}
											hide
										/>
										<YAxis
											tick={{ fill: "#8892a8", fontSize: 10 }}
											allowDecimals={false}
										/>
										<Tooltip
											contentStyle={{
												background: "#1a1f2e",
												border: "1px solid #2a3142",
												borderRadius: "8px",
												fontSize: "12px",
											}}
										/>
										<Area
											type="monotone"
											dataKey="count"
											stroke="#ef4444"
											fill="url(#alertGrad)"
											strokeWidth={2}
										/>
									</AreaChart>
								</ResponsiveContainer>
							</div>
						</div>
					</div>
				</div>

				<div className="card bg-base-200 shadow-sm">
					<div className="card-body space-y-4">
						<h2 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
							<IconSend size={18} className="text-primary" />
							Submit Transaction
						</h2>

						<div className="flex gap-2">
							<button
								className="btn btn-sm btn-outline btn-success flex-1 gap-1.5"
								onClick={() => setTxForm(LEGIT_PRESET)}
							>
								<IconCheckCircle size={16} />
								Load Legitimate
							</button>
							<button
								className="btn btn-sm btn-outline btn-error flex-1 gap-1.5"
								onClick={() => setTxForm(FRAUD_PRESET)}
							>
								<IconXmarkCircle size={16} />
								Load Fraud
							</button>
						</div>

						<div className="grid grid-cols-2 gap-2">
							<label className="flex flex-col gap-1">
								<span className="text-xs text-base-content/60">Time</span>
								<input
									type="number"
									step="any"
									className="input input-sm w-full font-mono"
									value={txForm.Time}
									onChange={(e) => handleFieldChange("Time", e.target.value)}
								/>
							</label>
							<label className="flex flex-col gap-1">
								<span className="text-xs text-base-content/60">Amount</span>
								<input
									type="number"
									step="any"
									className="input input-sm w-full font-mono"
									value={txForm.Amount}
									onChange={(e) => handleFieldChange("Amount", e.target.value)}
								/>
							</label>
						</div>

						<div className="grid grid-cols-4 gap-1.5 max-h-48 overflow-y-auto">
							{vFields.map((f) => (
								<label key={f} className="flex flex-col gap-0.5">
									<span className="text-[10px] text-base-content/50">{f}</span>
									<input
										type="number"
										step="any"
										className="input input-xs w-full font-mono"
										value={txForm[f]}
										onChange={(e) => handleFieldChange(f, e.target.value)}
									/>
								</label>
							))}
						</div>

						<button
							className="btn btn-primary btn-block gap-2"
							onClick={() => submitTransaction(txForm)}
							disabled={submitting}
						>
							{submitting ? (
								<span className="loading loading-spinner loading-sm" />
							) : (
								<>
									<IconSend size={18} />
									Send to API
								</>
							)}
						</button>

						{lastResult && (
							<div
								className={`alert ${
									lastResult.is_fraud ? "alert-error" : "alert-success"
								} alert-soft flex-col items-start gap-2`}
							>
								<div className="flex items-center justify-between w-full">
									<span className="badge badge-sm gap-1">
										{lastResult.is_fraud ? (
											<>
												<IconXmarkCircle size={14} />
												FRAUD
											</>
										) : (
											<>
												<IconCheckCircle size={14} />
												LEGITIMATE
											</>
										)}
									</span>
									<span className="text-xs text-base-content/60 font-mono">
										P={lastResult.probability.toFixed(4)} | T=
										{lastResult.threshold.toFixed(4)}
									</span>
								</div>
								{lastResult.top_features.length > 0 && (
									<div className="flex flex-col gap-0.5 w-full text-xs font-mono">
										{lastResult.top_features.map((f, i) => (
											<div key={i} className="flex justify-between">
												<span>{f.feature}</span>
												<span
													className={
														f.shap_value > 0 ? "text-error" : "text-info"
													}
												>
													{f.shap_value > 0 ? "+" : ""}
													{f.shap_value.toFixed(4)}
												</span>
											</div>
										))}
									</div>
								)}
							</div>
						)}
					</div>
				</div>

				<div className="card bg-base-200 shadow-sm">
					<div className="card-body space-y-4">
						<h2 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
							<IconCreditCard size={18} className="text-primary" />
							Transaction Feed
						</h2>
						<div className="flex-1 overflow-y-auto overflow-x-hidden max-h-[460px]">
							{transactions.length === 0 ? (
								<div className="flex items-center justify-center h-full text-base-content/40 text-sm text-center py-10">
									No transactions yet.
									<br />
									Submit one from the panel on the left, or run the simulator.
								</div>
							) : (
								<div className="flex flex-col gap-2">
									{transactions.map((item: Tx) => (
										<button
											key={item.id}
											className={`flex items-center justify-between rounded-box px-3 py-2.5 text-left transition-all hover:scale-[1.01] hover:shadow-md cursor-pointer ${
												item.is_fraud
													? "bg-error/3 border border-error/20"
													: "bg-base-300/50 border border-transparent"
											}`}
											onClick={() => setSelectedTx(item)}
										>
											<div className="flex flex-col gap-0.5 min-w-0">
												<span className="font-semibold text-sm truncate flex items-center gap-1.5">
													<IconDollar
														size={14}
														className="text-base-content/40"
													/>
													{item.amount.toFixed(2)}
												</span>
												<span className="text-[10px] text-base-content/80 flex items-center gap-1">
													<IconClock
														size={10}
														className="text-base-content/30"
													/>
													{fmtTime(item.time)} · #{item.id}
												</span>
											</div>
											<div className="flex items-center gap-2">
												<span className="text-xs text-base-content/50 font-mono">
													{(item.probability * 100).toFixed(1)}%
												</span>
												{item.is_fraud ? (
													<span className="badge badge-error badge-sm gap-1">
														<IconXmarkCircle size={12} />
														Fraud
													</span>
												) : (
													<span className="badge badge-success badge-sm gap-1">
														<IconCheckCircle size={12} />
														OK
													</span>
												)}
											</div>
										</button>
									))}
								</div>
							)}
							{transactions.length > 0 && hasMore && (
								<button
									className="btn btn-ghost btn-sm btn-block mt-3 gap-1.5"
									onClick={loadMore}
									disabled={loadingMore}
								>
									{loadingMore ? (
										<span className="loading loading-spinner loading-xs" />
									) : (
										<>
											<IconLayers size={16} />
											Load More
										</>
									)}
								</button>
							)}
						</div>
					</div>
				</div>
			</div>

			<div className="px-6 pb-6">
				<div className="card bg-base-200 shadow-sm">
					<div className="card-body space-y-4">
						<div className="flex items-center justify-between">
							<h2 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
								<IconBell size={18} className="text-error" />
								Recent Alerts
							</h2>
							{alerts.length > 0 && (
								<span className="badge badge-error badge-sm gap-1">
									<IconBell size={12} />
									{alerts.length} shown
								</span>
							)}
						</div>
						{alerts.length === 0 ? (
							<div className="text-base-content/40 text-sm text-center py-8">
								No alerts yet. Fraud alerts will appear here in real-time.
							</div>
						) : (
							<div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-3">
								{alerts.map((alert: Tx) => (
									<button
										key={alert.id}
										className="card bg-error/3 border border-error/20 hover:border-error/40 hover:shadow-md transition-all text-left cursor-pointer group"
										onClick={() => setSelectedTx(alert)}
									>
										<div className="card-body p-3 gap-2">
											<div className="flex items-center justify-between">
												<span className="text-lg font-bold text-error flex items-center gap-1">
													<IconDollar size={16} />
													{alert.amount.toFixed(2)}
												</span>
												<span className="badge badge-error badge-xs">
													{(alert.probability * 100).toFixed(1)}%
												</span>
											</div>
											<div className="flex items-center justify-between text-xs text-base-content/80">
												<span className="flex items-center gap-1">
													<IconClock size={10} />
													{fmtTime(alert.time)}
												</span>
												<span className="flex items-center gap-1 opacity-80 group-hover:opacity-100 transition-opacity">
													<IconEye size={12} />#{alert.id}
												</span>
											</div>
										</div>
									</button>
								))}
							</div>
						)}
					</div>
				</div>
			</div>

			<TxDetailModal tx={selectedTx} onClose={() => setSelectedTx(null)} />
		</div>
	);
}

export default App;
