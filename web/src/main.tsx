import React from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, RouterProvider } from "react-router-dom";
import { Toaster } from "sonner";
import App from "./App";
import Simulate from "./Simulate";
import { GlobalProvider } from "./store";
import "./index.css";

const router = createBrowserRouter([
	{
		path: "/",
		element: (
			<GlobalProvider>
				<App />
			</GlobalProvider>
		),
	},
	{
		path: "/simulate",
		element: (
			<GlobalProvider>
				<Simulate />
			</GlobalProvider>
		),
	},
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
	<React.StrictMode>
		<RouterProvider router={router} />
		<Toaster
			position="bottom-right"
			theme="dark"
			closeButton
			expand={false}
			toastOptions={{
				style: {
					fontFamily: "inherit",
					background: "rgba(220, 38, 38, 0.25)",
					backdropFilter: "blur(8px)",
				},
			}}
		/>
	</React.StrictMode>,
);
