import {
	IconSimulate,
	IconBolt,
	IconDatabase,
	IconLayers,
	IconShieldDollar,
	IconArrowRight,
	IconXmarkCircle,
	IconGlobe,
	IconCreditCard,
	IconDollar,
	IconClock,
	IconAngleDoubleRight,
} from "./icons";

const RISK_FACTORS = [
	{
		icon: IconCreditCard,
		label: "Card Not Present",
		contribution: "+15%",
		condition: "when card is absent",
	},
	{
		icon: IconBolt,
		label: "New Device",
		contribution: "+20%",
		condition: "first use on device",
	},
	{
		icon: IconGlobe,
		label: "Foreign Transaction",
		contribution: "+15%",
		condition: "outside home country",
	},
	{
		icon: IconClock,
		label: "Late Night (23:00–06:00)",
		contribution: "+15%",
		condition: "unusual hours",
	},
	{
		icon: IconDollar,
		label: "High Amount (>$500)",
		contribution: "+15%",
		condition: "large transaction",
	},
];

const MERCHANT_BASE_RISK = [
	{ label: "Grocery", risk: "5%" },
	{ label: "Restaurant", risk: "8%" },
	{ label: "Gas Station", risk: "10%" },
	{ label: "ATM", risk: "30%" },
	{ label: "Electronics", risk: "45%" },
	{ label: "Online", risk: "55%" },
];

const PIPELINE_STEPS = [
	{
		icon: IconSimulate,
		title: "Human Inputs",
		subtitle: "Scenario builder",
		items: ["Merchant type", "Amount", "Time of day", "Risk factors"],
		color: "text-primary",
		bg: "bg-primary/10",
		border: "border-primary/30",
	},
	{
		icon: IconBolt,
		title: "Risk Scoring",
		subtitle: "Weighted formula",
		items: ["Base merchant risk", "+ Factor penalties", "Caps at 95%"],
		color: "text-warning",
		bg: "bg-warning/10",
		border: "border-warning/30",
	},
	{
		icon: IconDatabase,
		title: "Dataset Sampling",
		subtitle: "Nearest neighbor",
		items: [
			"Fraud or legit pool",
			"Match by amount + time",
			"Pull real V1–V28",
		],
		color: "text-info",
		bg: "bg-info/10",
		border: "border-info/30",
	},
	{
		icon: IconLayers,
		title: "PCA Features",
		subtitle: "V1–V28 + Time + Amount",
		items: [
			"28 anonymized features",
			"From real ULB dataset",
			"Ready for model",
		],
		color: "text-secondary",
		bg: "bg-secondary/10",
		border: "border-secondary/30",
	},
	{
		icon: IconShieldDollar,
		title: "Model Prediction",
		subtitle: "XGBoost + SHAP",
		items: [
			"Fraud probability",
			"Threshold comparison",
			"Feature explanations",
		],
		color: "text-error",
		bg: "bg-error/10",
		border: "border-error/30",
	},
];

export function PipelineDiagram() {
	return (
		<div className="card bg-base-200 shadow-sm">
			<div className="card-body space-y-6">
				<div className="space-y-1">
					<h2 className="text-sm font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
						<IconSimulate size={18} className="text-primary" />
						How Simulation Works
					</h2>
					<p className="text-sm text-base-content/60">
						The ULB dataset stores transactions as 28 anonymized PCA features
						(V1–V28). Since these have no human-readable meaning, the simulator
						works in reverse: it takes your scenario, computes a risk score,
						finds the closest real transaction in the dataset, and extracts its
						PCA features for the model.
					</p>
				</div>

				<div className="flex flex-col lg:flex-row items-stretch gap-2">
					{PIPELINE_STEPS.map((step, i) => {
						const StepIcon = step.icon;
						return (
							<div
								key={i}
								className="flex flex-col lg:flex-row items-center gap-2 flex-1"
							>
								<div
									className={`flex-1 rounded-box border-2 ${step.border} ${step.bg} p-4 space-y-2 w-full`}
								>
									<div className="flex items-center gap-2">
										<StepIcon size={22} className={step.color} />
										<div>
											<div className={`text-sm font-bold ${step.color}`}>
												{step.title}
											</div>
											<div className="text-[10px] text-base-content/50 uppercase tracking-wide">
												{step.subtitle}
											</div>
										</div>
									</div>
									<div className="flex flex-col gap-1">
										{step.items.map((item, j) => (
											<div
												key={j}
												className="text-xs text-base-content/70 flex items-center gap-1.5"
											>
												<IconAngleDoubleRight
													size={12}
													className="text-base-content/50"
												/>
												{item}
											</div>
										))}
									</div>
								</div>
								{i < PIPELINE_STEPS.length - 1 && (
									<div className="flex lg:flex-col items-center justify-center px-1">
										<IconArrowRight
											size={20}
											className="text-base-content/30 rotate-90 lg:rotate-0"
										/>
									</div>
								)}
							</div>
						);
					})}
				</div>

				<div className="grid grid-cols-1 md:grid-cols-2 gap-4">
					<div className="space-y-3">
						<h3 className="text-xs font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
							<IconBolt size={14} className="text-warning" />
							Merchant Base Risk
						</h3>
						<div className="space-y-1.5">
							{MERCHANT_BASE_RISK.map((m) => (
								<div
									key={m.label}
									className="flex items-center justify-between bg-base-300/50 rounded px-3 py-1.5"
								>
									<span className="text-sm">{m.label}</span>
									<div className="flex items-center gap-2 flex-1 ml-3">
										<div className="flex-1 h-2 bg-base-300 rounded-full overflow-hidden">
											<div
												className={`h-full rounded-full ${
													parseFloat(m.risk) > 30
														? "bg-error"
														: parseFloat(m.risk) > 10
															? "bg-warning"
															: "bg-success"
												}`}
												style={{ width: `${parseFloat(m.risk) * 1.5}%` }}
											/>
										</div>
										<span className="text-xs font-mono w-10 text-right">
											{m.risk}
										</span>
									</div>
								</div>
							))}
						</div>
					</div>

					<div className="space-y-3">
						<h3 className="text-xs font-semibold uppercase tracking-wider text-base-content/50 flex items-center gap-2">
							<IconBolt size={14} className="text-warning" />
							Risk Factor Penalties
						</h3>
						<div className="space-y-1.5">
							{RISK_FACTORS.map((f) => {
								const FactorIcon = f.icon;
								return (
									<div
										key={f.label}
										className="flex items-center gap-3 bg-base-300/50 rounded px-3 py-1.5"
									>
										<FactorIcon size={18} className="text-base-content/50" />
										<div className="flex-1 min-w-0">
											<div className="text-sm">{f.label}</div>
											<div className="text-[10px] text-base-content/40">
												{f.condition}
											</div>
										</div>
										<span className="text-xs font-mono text-error font-semibold">
											{f.contribution}
										</span>
									</div>
								);
							})}
						</div>
					</div>
				</div>

				<div className="rounded-box border border-base-300 bg-base-300/30 p-4 space-y-2">
					<h3 className="text-xs font-semibold uppercase tracking-wider text-base-content/50">
						Example: Online purchase at 02:00, $800, new device, foreign
					</h3>
					<div className="flex flex-wrap items-center gap-2 text-xs">
						<span className="badge badge-sm gap-1">
							<IconGlobe size={12} />
							Online 55%
						</span>
						<IconArrowRight size={14} className="text-base-content/30" />
						<span className="badge badge-sm badge-warning gap-1">
							<IconBolt size={12} />
							+20% new device
						</span>
						<span className="badge badge-sm badge-warning gap-1">
							<IconBolt size={12} />
							+15% foreign
						</span>
						<span className="badge badge-sm badge-warning gap-1">
							<IconBolt size={12} />
							+15% late night
						</span>
						<span className="badge badge-sm badge-warning gap-1">
							<IconBolt size={12} />
							+15% high amount
						</span>
						<IconArrowRight size={14} className="text-base-content/30" />
						<span className="badge badge-sm badge-error">
							Risk = 95% (capped)
						</span>
						<IconArrowRight size={14} className="text-base-content/30" />
						<span className="badge badge-sm badge-info gap-1">
							<IconDatabase size={12} />
							Sample from fraud pool
						</span>
						<IconArrowRight size={14} className="text-base-content/30" />
						<span className="badge badge-sm badge-error gap-1">
							<IconXmarkCircle size={12} />
							Fraud detected
						</span>
					</div>
				</div>
			</div>
		</div>
	);
}
