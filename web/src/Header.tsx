import { Link, useLocation } from "react-router-dom";
import { useGlobal } from "./store";
import { IconDashboard, IconSimulate } from "./icons";
import { VigilLogo } from "./VigilLogo";

export function Header() {
	const { connected } = useGlobal();
	const location = useLocation();

	return (
		<div className="navbar bg-base-200 shadow-sm px-6 gap-4">
			<div className="flex-1 flex items-center gap-2.5">
				<VigilLogo size={32} />
				<div className="flex flex-col leading-none">
					<h1 className="text-3xl tracking-[0.15em] montserrat-regular">
						VIGIL
					</h1>
					<span className="text-[10px] font-medium text-base-content/40 tracking-widest uppercase">
						Fraud Detection
					</span>
				</div>
			</div>

			<div className="flex-none">
				<div role="tablist" className="tabs tabs-lift tabs-sm">
					<Link
						to="/"
						role="tab"
						className={`tab gap-1.5 ${location.pathname === "/" ? "tab-active" : ""}`}
					>
						<IconDashboard size={16} />
						Dashboard
					</Link>
					<Link
						to="/simulate"
						role="tab"
						className={`tab gap-1.5 ${
							location.pathname === "/simulate" ? "tab-active" : ""
						}`}
					>
						<IconSimulate size={16} />
						Simulate
					</Link>
				</div>
			</div>

			<div className="flex-none">
				<div className="flex items-center gap-2">
					<span
						className={`status status-sm ${
							connected ? "status-success" : "status-error"
						}`}
					/>
					<span className="text-sm text-base-content/60">
						{connected ? "Live" : "Disconnected"}
					</span>
				</div>
			</div>
		</div>
	);
}
