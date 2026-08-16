type LogoProps = {
	size?: number;
	className?: string;
};

export function VigilLogo({ size = 32, className = "" }: LogoProps) {
	return (
		<div
			className={`relative inline-flex items-center justify-center ${className}`}
			style={{ width: size, height: size }}
		>
			<svg
				width={size}
				height={size}
				viewBox="0 0 48 48"
				fill="none"
				xmlns="http://www.w3.org/2000/svg"
			>
				<defs>
					<linearGradient id="cardGrad" x1="8" y1="14" x2="40" y2="34">
						<stop offset="0%" stopColor="oklch(0.55 0.13 200)" />
						<stop offset="100%" stopColor="oklch(0.45 0.15 210)" />
					</linearGradient>
					<linearGradient id="scanGrad" x1="20" y1="18" x2="28" y2="30">
						<stop offset="0%" stopColor="oklch(0.72 0.17 150)" />
						<stop offset="100%" stopColor="oklch(0.65 0.20 160)" />
					</linearGradient>
				</defs>

				<rect
					x="6"
					y="12"
					width="36"
					height="24"
					rx="4"
					fill="url(#cardGrad)"
				/>

				<rect x="10" y="17" width="8" height="6" rx="1" fill="oklch(0.85 0.04 80)" opacity="0.9" />

				<rect x="10" y="27" width="14" height="2" rx="1" fill="oklch(0.95 0.01 0)" opacity="0.4" />
				<rect x="10" y="31" width="9" height="2" rx="1" fill="oklch(0.95 0.01 0)" opacity="0.25" />

				<circle cx="32" cy="24" r="7" fill="oklch(0.20 0.02 200)" opacity="0.6" />
				<circle cx="32" cy="24" r="5" fill="url(#scanGrad)" />
				<circle cx="32" cy="24" r="2.5" fill="oklch(0.15 0.02 200)" />
				<circle cx="33" cy="23" r="0.8" fill="oklch(0.95 0.02 200)" opacity="0.9" />

				<path
					d="M25 24 Q32 19 39 24 Q32 29 25 24 Z"
					stroke="oklch(0.72 0.17 150)"
					strokeWidth="1.2"
					fill="none"
					opacity="0.5"
				/>
			</svg>

			<span
				className="absolute rounded-full"
				style={{
					width: 6,
					height: 6,
					top: size * 0.42,
					right: size * 0.12,
					background: "oklch(0.72 0.17 150)",
					boxShadow: "0 0 6px oklch(0.72 0.17 150)",
					animation: "vigil-pulse 2s ease-in-out infinite",
				}}
			/>
		</div>
	);
}
