
import tkinter as tk
from tkinter import ttk, messagebox
import requests

# --- Colors and Styles ---
COLOR_HEADER = "#222244"
COLOR_TRADE = "#e0ffe0"
COLOR_SETUP = "#ffe0e0"
COLOR_ML = "#e0e0ff"
COLOR_BUY = "#00b300"
COLOR_SELL = "#d11a1a"

def main():
	root = tk.Tk()
	root.title("PYTHON_EA Dashboard")
	root.geometry("1100x700")
	root.configure(bg="#222233")

	# --- AI and Goal Section ---
	ai_goal_frame = tk.Frame(root, bg="#222233")
	ai_goal_frame.pack(fill=tk.X, pady=8)
	ai_icon_big = tk.Label(ai_goal_frame, text="🤖", font=("Arial", 60), bg="#222233", fg="#00e6e6")
	ai_icon_big.pack(side=tk.LEFT, padx=20)
	goal_frame = tk.Frame(ai_goal_frame, bg="#222233")
	goal_frame.pack(side=tk.LEFT, padx=10)
	goal_title = tk.Label(goal_frame, text="Goal: Cover Monthly Expenses", font=("Arial", 16, "bold"), bg="#222233", fg="#fff")
	goal_title.pack(anchor="w")
	goal_progress = tk.DoubleVar()
	goal_bar = ttk.Progressbar(goal_frame, orient="horizontal", length=350, mode="determinate", variable=goal_progress, maximum=1.0)
	goal_bar.pack(fill=tk.X, pady=4)
	goal_label = tk.Label(goal_frame, text="", font=("Arial", 13), bg="#222233", fg="#fff")
	goal_label.pack(anchor="w")
	goal_message = tk.Label(goal_frame, text="", font=("Arial", 13, "italic"), bg="#222233", fg="#00ff99")
	goal_message.pack(anchor="w", pady=2)

	header = tk.Label(root, text="PYTHON_EA Dashboard", font=("Arial", 20, "bold"), bg="#222233", fg="#fff")
	header.pack(pady=2)

	# --- Tabbed Interface ---
	tabs = ttk.Notebook(root)
	tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
	frame = tk.Frame(tabs, bg="#222233")
	tabs.add(frame, text="Overview")
	trade_hist_text = tk.Text(tabs, font=("Consolas", 11), height=12, bg="#181825", fg="#fff")
	tabs.add(trade_hist_text, text="Trade History")
	news_text = tk.Text(tabs, font=("Consolas", 11), height=6, bg="#181825", fg="#fff")
	tabs.add(news_text, text="News & Sentiment")
	risk_text = tk.Text(tabs, font=("Consolas", 11), height=6, bg="#181825", fg="#fff")
	tabs.add(risk_text, text="Risk Metrics")
	perf_text = tk.Text(tabs, font=("Consolas", 11), height=6, bg="#181825", fg="#fff")
	tabs.add(perf_text, text="Performance")
	watchlist_text = tk.Text(tabs, font=("Consolas", 11), height=6, bg="#181825", fg="#fff")
	tabs.add(watchlist_text, text="Watchlist")
	strat_text = tk.Text(tabs, font=("Consolas", 11), height=6, bg="#181825", fg="#fff")
	tabs.add(strat_text, text="Strategy Insights")

	active_trades_text = tk.Text(frame, font=("Consolas", 11), height=10, bg="#181825", fg="#fff")
	active_trades_text.pack(fill=tk.X, pady=5)
	active_trades_text.insert(tk.END, "Active Trades\n", "header")
	active_trades_text.tag_config("header", background=COLOR_HEADER, foreground="#fff", font=("Arial", 12, "bold"))

	setups_text = tk.Text(frame, font=("Consolas", 11), height=8, bg="#252525", fg="#fff")
	setups_text.pack(fill=tk.X, pady=5)
	setups_text.insert(tk.END, "Detected Setups\n", "header")
	setups_text.tag_config("header", background=COLOR_HEADER, foreground="#fff", font=("Arial", 12, "bold"))

	ai_frame = tk.Frame(root, bg="#222233")
	ai_frame.pack(pady=5)
	ai_icon = tk.Label(ai_frame, text="🤖", font=("Arial", 40), bg="#222233", fg="#fff")
	ai_icon.pack(side=tk.LEFT, padx=10)
	ai_status = tk.Label(ai_frame, text="AI Status: ...", font=("Arial", 14), bg="#222233", fg="#fff")
	ai_status.pack(side=tk.LEFT)

	ml_text = tk.Text(frame, font=("Consolas", 11), height=4, bg="#222244", fg="#fff")
	ml_text.pack(fill=tk.X, pady=5)
	ml_text.insert(tk.END, "ML Updates\n", "header")
	ml_text.tag_config("header", background=COLOR_HEADER, foreground="#fff", font=("Arial", 12, "bold"))

	pred_text = tk.Text(frame, font=("Consolas", 11), height=4, bg="#1a1a2a", fg="#fff")
	pred_text.pack(fill=tk.X, pady=5)
	pred_text.insert(tk.END, "Predictions\n", "header")
	pred_text.tag_config("header", background=COLOR_HEADER, foreground="#fff", font=("Arial", 12, "bold"))

	session_text = tk.Text(frame, font=("Consolas", 11), height=4, bg="#23233a", fg="#fff")
	session_text.pack(fill=tk.X, pady=5)
	session_text.insert(tk.END, "Sessions\n", "header")
	session_text.tag_config("header", background=COLOR_HEADER, foreground="#fff", font=("Arial", 12, "bold"))

	setupd_text = tk.Text(frame, font=("Consolas", 11), height=3, bg="#2a2a3a", fg="#fff")
	setupd_text.pack(fill=tk.X, pady=5)
	setupd_text.insert(tk.END, "Setup Details\n", "header")
	setupd_text.tag_config("header", background=COLOR_HEADER, foreground="#fff", font=("Arial", 12, "bold"))

	def refresh_data():
		try:
			r = requests.get("http://127.0.0.1:5000/stats")
			data = r.json()

			# AI robot big icon and motivational message
			ai = data.get("ai_robot", {})
			ai_icon_big.config(text=ai.get("mood", "🤖"))

			# Goal progress bar and message
			goals = data.get("goals", {})
			monthly = goals.get("monthly_expenses", 2000)
			pnl = goals.get("current_pnl", 0)
			progress = min(pnl / monthly, 1.0) if monthly else 0
			goal_progress.set(progress)
			goal_label.config(text=f"Progress: R{pnl} / R{monthly}")
			if progress >= 1.0:
				goal_message.config(text=goals.get("goal_message", "Goal reached!"), fg="#00ff99")
			else:
				goal_message.config(text="Keep going!", fg="#ffb366")

			# Animated status (simple)
			ai_status.after(500, lambda: ai_status.config(fg="#00ff00" if ai_status.cget("fg")=="#fff" else "#fff"))

			# Notifications/alerts
			if any(abs(t.get("profit",0)) > 100 for t in data.get("active_trades", [])):
				messagebox.showinfo("Big Trade Alert!", "A trade has profit/loss over 100!")

			# Trade History
			trade_hist_text.delete("1.0", tk.END)
			trade_hist_text.insert(tk.END, "Symbol   Type   Entry     Exit      PnL   Duration  Result\n", "header")
			for t in data.get("trade_history", []):
				line = f"{t['symbol']:<8}{t['type']:<7}{t['entry']:<9}{t['exit']:<9}{t['pnl']:<6}{t['duration']:<9}{t['result']}\n"
				trade_hist_text.insert(tk.END, line)
			trade_hist_text.tag_config("header", font=("Arial", 12, "bold"), foreground="#00e6e6")

			# News & Sentiment
			news_text.delete("1.0", tk.END)
			news_text.insert(tk.END, "News Feed\n", "header")
			for n in data.get("news_feed", []):
				news_text.insert(tk.END, f"{n['symbol']}: {n['headline']}\n")
			news_text.insert(tk.END, "\nSentiment\n", "header")
			for sym, sent in data.get("sentiment", {}).items():
				news_text.insert(tk.END, f"{sym}: {sent}\n")
			news_text.tag_config("header", font=("Arial", 12, "bold"), foreground="#ffb366")

			# Risk Metrics
			risk = data.get("risk_metrics", {})
			risk_text.delete("1.0", tk.END)
			risk_text.insert(tk.END, f"Current Drawdown: {risk.get('current_drawdown','N/A')}%\nMax Drawdown: {risk.get('max_drawdown','N/A')}%\nRisk/Trade: {risk.get('risk_per_trade','N/A')}%\n")
			risk_text.insert(tk.END, "Exposure by Symbol:\n")
			for sym, exp in risk.get("exposure", {}).items():
				risk_text.insert(tk.END, f"  {sym}: {exp*100:.1f}%\n")

			# Performance Analytics
			perf = data.get("performance", {})
			perf_text.delete("1.0", tk.END)
			perf_text.insert(tk.END, f"Sharpe Ratio: {perf.get('sharpe','N/A')}\nExpectancy: {perf.get('expectancy','N/A')}\nAvg Hold: {perf.get('avg_hold','N/A')}\nBest Period: {perf.get('best_period','N/A')}\nWorst Period: {perf.get('worst_period','N/A')}\n")

			# Watchlist
			watchlist_text.delete("1.0", tk.END)
			watchlist_text.insert(tk.END, "Symbol Watchlist\n", "header")
			for sym in data.get("watchlist", []):
				watchlist_text.insert(tk.END, f"{sym}\n")
			watchlist_text.tag_config("header", font=("Arial", 12, "bold"), foreground="#b3b3ff")

			# Strategy Insights
			strat_text.delete("1.0", tk.END)
			strat_text.insert(tk.END, "Strategy   Trades  WinRate  AvgRR\n", "header")
			for s in data.get("strategy_insights", []):
				strat_text.insert(tk.END, f"{s['name']:<10}{s['trades']:<8}{s['win_rate']:<9}{s['avg_rr']}\n")
			strat_text.tag_config("header", font=("Arial", 12, "bold"), foreground="#b3b3ff")

			# Active Trades
			active_trades_text.delete("2.0", tk.END)
			for trade in data.get("active_trades", []):
				color = COLOR_BUY if trade.get("type", "").lower() == "buy" else COLOR_SELL
				line = f"{trade['symbol']:<10}{trade['type'].upper():<6}Lot:{trade['lot']:<5}Entry:{trade['entry']:<8.5f} SL:{trade['sl']:<8.5f} TP:{trade['tp']:<8.5f} Profit:{trade['profit']}\n"
				active_trades_text.insert(tk.END, line, (color,))
				active_trades_text.tag_config(color, foreground=color)

			# Detected Setups
			setups_text.delete("2.0", tk.END)
			for setup in data.get("detected_setups", []):
				conf = setup.get("confidence", 0)
				conf_color = COLOR_BUY if setup.get("signal", "").lower() == "buy" else COLOR_SELL
				line = f"{setup['symbol']:<10}{setup['signal'].upper():<6}Conf:{conf:<5.2f}Entry:{setup['entry']:<8.5f} SL:{setup['sl']:<8.5f} TP:{setup['tp']:<8.5f}\n"
				setups_text.insert(tk.END, line, (conf_color,))
				setups_text.tag_config(conf_color, foreground=conf_color)

			# ML Updates
			ml_text.delete("2.0", tk.END)
			ml = data.get("ml_updates", {})
			if ml:
				ml_line = f"Accuracy: {ml.get('accuracy', 'N/A')}\nLast Train: {ml.get('last_train', 'N/A')}\nTotal Trades: {ml.get('total_trades', 'N/A')}\nWin Rate: {ml.get('win_rate', 'N/A')}\nAvg RR: {ml.get('avg_rr', 'N/A')}\nBest: {ml.get('best_symbol', 'N/A')}  Worst: {ml.get('worst_symbol', 'N/A')}\n"
				ml_text.insert(tk.END, ml_line, ("ml",))
				ml_text.tag_config("ml", foreground=COLOR_ML)
			else:
				ml_text.insert(tk.END, "No ML updates.\n", ("ml",))
				ml_text.tag_config("ml", foreground=COLOR_ML)

			# Predictions
			pred_text.delete("2.0", tk.END)
			preds = data.get("predictions", {})
			if preds:
				for period in ["daily", "weekly", "monthly"]:
					p = preds.get(period, {})
					line = f"{period.title():<8} PnL: {p.get('expected_pnl','N/A'):<6} Trades: {p.get('expected_trades','N/A'):<4} WinRate: {p.get('expected_winrate','N/A')}\n"
					pred_text.insert(tk.END, line, ("pred",))
				pred_text.tag_config("pred", foreground="#00e6e6")
			else:
				pred_text.insert(tk.END, "No predictions.\n", ("pred",))
				pred_text.tag_config("pred", foreground="#00e6e6")

			# Sessions
			session_text.delete("2.0", tk.END)
			sessions = data.get("sessions", [])
			if sessions:
				for s in sessions:
					line = f"{s['name']:<8} Trades: {s['trades']:<3} PnL: {s['pnl']:<5} Best: {s['best_symbol']}\n"
					session_text.insert(tk.END, line, ("sess",))
				session_text.tag_config("sess", foreground="#ffb366")
			else:
				session_text.insert(tk.END, "No session data.\n", ("sess",))
				session_text.tag_config("sess", foreground="#ffb366")

			# Setup details
			setupd_text.delete("2.0", tk.END)
			setupd = data.get("setup_details", {})
			if setupd:
				line = f"Most Common: {setupd.get('most_common','N/A')}\nLast Setup: {setupd.get('last_setup','N/A')} {setupd.get('last_symbol','')} (Conf: {setupd.get('last_confidence','N/A')})\n"
				setupd_text.insert(tk.END, line, ("setupd",))
				setupd_text.tag_config("setupd", foreground="#b3b3ff")
			else:
				setupd_text.insert(tk.END, "No setup details.\n", ("setupd",))
				setupd_text.tag_config("setupd", foreground="#b3b3ff")

			# AI Robot status
			ai = data.get("ai_robot", {})
			if ai:
				ai_icon.config(text=ai.get("mood", "🤖"))
				ai_status.config(text=f"AI Status: {ai.get('status','offline').capitalize()} | {ai.get('message','')}")
			else:
				ai_icon.config(text="🤖")
				ai_status.config(text="AI Status: offline")

		except Exception as e:
			active_trades_text.delete("2.0", tk.END)
			active_trades_text.insert(tk.END, f"Error fetching data: {e}\n", ("error",))
			active_trades_text.tag_config("error", foreground="red")

		root.after(3000, refresh_data)

	refresh_data()
	root.mainloop()

if __name__ == "__main__":
	main()
