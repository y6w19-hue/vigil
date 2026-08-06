import { useState } from "react";
import { Header } from "./Header";
import { PipelineDiagram } from "./PipelineDiagram";
import {
	IconSimulate,
	IconSend,
	IconCheckCircle,
	IconXmarkCircle,
	IconDollar,
	IconClock,
	IconChart,
	IconLayers,
	IconShieldCheck,
	IconShieldDollar,
	IconBolt,
	IconGlobe,
	IconCreditCard,
	IconBasket,
	IconUtensils,
	IconFuel,
	IconLaptop,
	IconWww,
	IconDollarCircle,
} from "./icons";

const API_URL = `http://${window.location.hostname}:8000`;

interface ScenarioResult {
	transaction: Record<string, number>;
	risk_score: number;
	sampled_from: string;
	scenario: {
		merchant: string;
		amount: number;
		hour: number;
		card_present: boolean;
		new_device: boolean;
		foreign: boolean;
	};
	prediction: {
		is_fraud: boolean;
		probability: number;
		threshold: number;
		top_features: { feature: string; shap_value: number }[];
	};
}

const MERCHANTS = [
	{ value: "grocery", label: "Grocery", icon: IconBasket },
	{ value: "restaurant", label: "Restaurant", icon: IconUtensils },
	{ value: "gas", label: "Gas Station", icon: IconFuel },
	{ value: "electronics", label: "Electronics", icon: IconLaptop },
	{ value: "online", label: "Online", icon: IconWww },
	{ value: "atm", label: "ATM", icon: IconDollarCircle },
];

function Simulate() {
	const [merchant, setMerchant] = useState("grocery");
	const [amount, setAmount] = useState(50);
	const [hour, setHour] = useState(14);
	const [cardPresent, setCardPresent] = useState(true);
	const [newDevice, setNewDevice] = useState(false);
	const [foreign, setForeign] = useState(false);
	const [loading, setLoading] = useState(false);
	const [result, setResult] = useState<ScenarioResult | null>(null);
	const [error, setError] = useState<string | null>(null);

	const submit = async () => {
		setLoading(true);
		setError(null);
		try {
			const resp = await fetch(`${API_URL}/scenario`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					merchant,
					amount,
					hour,
					card_present: cardPresent,
					new_device: newDevice,
					foreign,
				}),
			});
			const data = await resp.json();
			if (data.error) {
				setError(data.error);
			} else {
				setResult(data);
			}
		} catch {
			setError("Failed to connect to API");
		} finally {
			setLoading(false);
		}
	};

	const hourLabel = `${String(hour).padStart(2, "0")}:00`;

	return (
		<div className="min-h-screen flex flex-col">
			<Header />

			<div className="grid grid-cols-1 lg:grid-cols-2 gap-4 p-6 flex-1 items-stretch">
				<div className="card bg-base-200 shadow-sm">
					<div className="card-body space-y-5">
						<div className="space-y-2">
							<h2 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
								<IconSimulate size={18} className="text-primary" />
								Transaction Scenario
							</h2>
							<p className="text-sm text-base-content/60 leading-relaxed">
								Build a realistic credit card transaction from human-readable
								attributes. The system finds a matching real transaction from
								the dataset, extracts its PCA features (V1–V28), and runs it
								through the fraud detection model.
							</p>
						</div>

						<div className="space-y-4 flex flex-col">
							<span className="text-sm font-medium">Merchant Type</span>
							<div className="grid grid-cols-3 gap-3">
								{MERCHANTS.map((m) => {
									const MerchantIcon = m.icon;
									return (
										<button
											key={m.value}
											className={`flex flex-col items-center gap-2 p-4 rounded-box border-2 transition-all ${
												merchant === m.value
													? "border-primary bg-primary/10 text-primary"
													: "border-base-300 bg-base-300/30 hover:border-base-content/30"
											}`}
											onClick={() => setMerchant(m.value)}
										>
											<MerchantIcon size={28} />
											<span className="text-xs font-medium">{m.label}</span>
										</button>
									);
								})}
							</div>
						</div>

						<div className="space-y-2">
							<div className="flex items-center justify-between">
								<span className="text-sm font-medium flex items-center gap-1.5">
									<IconDollar size={14} className="text-base-content/40" />
									Amount
								</span>
								<span className="text-sm font-mono text-primary">
									${amount.toFixed(2)}
								</span>
							</div>
							<input
								type="range"
								min={1}
								max={2000}
								step={1}
								value={amount}
								onChange={(e) => setAmount(Number(e.target.value))}
								className="range range-primary range-xs"
							/>
							<div className="flex justify-between text-[10px] text-base-content/40">
								<span>$1</span>
								<span>$500</span>
								<span>$1000</span>
								<span>$2000</span>
							</div>
						</div>

						<div className="space-y-2">
							<div className="flex items-center justify-between">
								<span className="text-sm font-medium flex items-center gap-1.5">
									<IconClock size={14} className="text-base-content/40" />
									Time of Day
								</span>
								<span className="text-sm font-mono text-primary">
									{hourLabel}
								</span>
							</div>
							<input
								type="range"
								min={0}
								max={23}
								step={1}
								value={hour}
								onChange={(e) => setHour(Number(e.target.value))}
								className="range range-primary range-xs"
							/>
							<div className="flex justify-between text-[10px] text-base-content/40">
								<span>00:00</span>
								<span>06:00</span>
								<span>12:00</span>
								<span>18:00</span>
								<span>23:00</span>
							</div>
						</div>

						<div className="space-y-2">
							<span className="text-sm font-medium flex items-center gap-1.5">
								<IconBolt size={14} className="text-warning" />
								Risk Factors
							</span>
							<div className="flex flex-col gap-2">
								<div
									className={`flex items-center justify-between p-3 rounded-box border ${
										!cardPresent
											? "border-error/30 bg-error/5"
											: "border-base-300"
									}`}
								>
									<label className="flex items-center gap-3 cursor-pointer">
										<input
											type="checkbox"
											className="toggle toggle-sm"
											checked={cardPresent}
											onChange={(e) => setCardPresent(e.target.checked)}
										/>
										<span className="text-sm flex items-center gap-1.5">
											<IconCreditCard
												size={14}
												className="text-base-content/40"
											/>
											Card Present
										</span>
									</label>
									<span className="text-xs text-base-content/40">
										{cardPresent ? "Physical card" : "Card not present"}
									</span>
								</div>

								<div
									className={`flex items-center justify-between p-3 rounded-box border ${
										newDevice ? "border-error/30 bg-error/5" : "border-base-300"
									}`}
								>
									<label className="flex items-center gap-3 cursor-pointer">
										<input
											type="checkbox"
											className="toggle toggle-sm toggle-error"
											checked={newDevice}
											onChange={(e) => setNewDevice(e.target.checked)}
										/>
										<span className="text-sm flex items-center gap-1.5">
											<IconBolt size={14} className="text-base-content/40" />
											New Device
										</span>
									</label>
									<span className="text-xs text-base-content/40">
										{newDevice ? "First time on this device" : "Known device"}
									</span>
								</div>

								<div
									className={`flex items-center justify-between p-3 rounded-box border ${
										foreign ? "border-error/30 bg-error/5" : "border-base-300"
									}`}
								>
									<label className="flex items-center gap-3 cursor-pointer">
										<input
											type="checkbox"
											className="toggle toggle-sm toggle-error"
											checked={foreign}
											onChange={(e) => setForeign(e.target.checked)}
										/>
										<span className="text-sm flex items-center gap-1.5">
											<IconGlobe size={14} className="text-base-content/40" />
											Foreign Transaction
										</span>
									</label>
									<span className="text-xs text-base-content/40">
										{foreign ? "Outside home country" : "Domestic"}
									</span>
								</div>
							</div>
						</div>

						<button
							className="btn btn-primary btn-block gap-2"
							onClick={submit}
							disabled={loading}
						>
							{loading ? (
								<>
									<span className="loading loading-spinner loading-sm" />
									Processing...
								</>
							) : (
								<>
									<IconSend size={18} />
									Run Fraud Check
								</>
							)}
						</button>

						{error && (
							<div className="alert alert-error alert-soft gap-2">
								<IconXmarkCircle size={18} />
								<span>{error}</span>
							</div>
						)}
					</div>
				</div>

				{result && (
					<div className="card bg-base-200 shadow-sm">
						<div className="card-body space-y-4">
							<h2 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
								{result.prediction.is_fraud ? (
									<IconShieldDollar size={18} className="text-error" />
								) : (
									<IconShieldCheck size={18} className="text-success" />
								)}
								Detection Result
							</h2>

							<div
								className={`alert ${
									result.prediction.is_fraud ? "alert-error" : "alert-success"
								} alert-soft gap-2`}
							>
								<div className="flex items-center justify-between w-full">
									<span className="font-bold flex items-center gap-2">
										{result.prediction.is_fraud ? (
											<>
												<IconXmarkCircle size={20} />
												FRAUD DETECTED
											</>
										) : (
											<>
												<IconCheckCircle size={20} />
												LEGITIMATE
											</>
										)}
									</span>
									<span className="text-xs font-mono ml-2">
										P={result.prediction.probability.toFixed(4)} | T=
										{result.prediction.threshold.toFixed(4)}
									</span>
								</div>
							</div>

							<div className="bg-base-300/50 rounded-box p-4 space-y-2">
								<div className="flex items-center gap-3">
									<span className="text-xs text-base-content/50 uppercase tracking-wide w-24">
										Risk Score
									</span>
									<progress
										className="progress progress-warning flex-1"
										value={result.risk_score * 100}
										max={100}
									/>
									<span className="text-sm font-semibold">
										{(result.risk_score * 100).toFixed(1)}%
									</span>
								</div>
								<div className="flex items-center gap-3">
									<span className="text-xs text-base-content/50 uppercase tracking-wide w-24">
										Sampled From
									</span>
									<span
										className={`text-sm font-medium ${
											result.sampled_from === "fraud"
												? "text-error"
												: "text-success"
										}`}
									>
										{result.sampled_from === "fraud"
											? "Real fraud transaction"
											: "Real legitimate transaction"}
									</span>
								</div>
								<div className="flex items-center gap-3">
									<span className="text-xs text-base-content/50 uppercase tracking-wide w-24">
										Scenario
									</span>
									<span className="text-sm">
										{
											MERCHANTS.find(
												(m) => m.value === result.scenario.merchant,
											)?.label
										}{" "}
										· ${result.scenario.amount.toFixed(2)} ·{" "}
										{String(result.scenario.hour).padStart(2, "0")}:00
									</span>
								</div>
							</div>

							{result.prediction.top_features.length > 0 && (
								<div className="space-y-3">
									<h3 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
										<IconChart size={16} className="text-primary" />
										SHAP Feature Contributions
									</h3>
									<div className="flex flex-col gap-2">
										{result.prediction.top_features.map((f, i) => {
											const maxVal = Math.max(
												...result.prediction.top_features.map((x) =>
													Math.abs(x.shap_value),
												),
											);
											const widthPct = (Math.abs(f.shap_value) / maxVal) * 100;
											return (
												<div key={i} className="flex items-center gap-3">
													<span className="text-xs font-mono font-semibold w-10">
														{f.feature}
													</span>
													<div className="flex-1 h-4 bg-base-300 rounded overflow-hidden">
														<div
															className={`h-full rounded transition-all duration-500 ${
																f.shap_value > 0 ? "bg-error/60" : "bg-info/60"
															}`}
															style={{ width: `${widthPct}%` }}
														/>
													</div>
													<span
														className={`text-xs font-mono w-16 text-right ${
															f.shap_value > 0 ? "text-error" : "text-info"
														}`}
													>
														{f.shap_value > 0 ? "+" : ""}
														{f.shap_value.toFixed(4)}
													</span>
												</div>
											);
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

							<div className="space-y-3">
								<h3 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
									<IconLayers size={16} className="text-primary" />
									PCA Features (V1–V28){" "}
									<span className="text-xs normal-case text-base-content/40">
										from real dataset
									</span>
								</h3>
								<div className="grid grid-cols-4 gap-1.5">
									{Array.from({ length: 28 }, (_, i) => `V${i + 1}`).map(
										(v) => (
											<div
												key={v}
												className="bg-base-300/50 rounded p-2 border border-base-300"
											>
												<div className="text-[10px] text-base-content/50 font-semibold">
													{v}
												</div>
												<div className="text-xs font-mono">
													{result.transaction[v]?.toFixed(4) ?? "—"}
												</div>
											</div>
										),
									)}
								</div>
							</div>
						</div>
					</div>
				)}
			</div>

			<div className="px-6 pb-6">
				<PipelineDiagram />
			</div>
		</div>
	);
}

export default Simulate;
