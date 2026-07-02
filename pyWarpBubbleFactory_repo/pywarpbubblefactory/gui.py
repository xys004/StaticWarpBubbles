"""
pywarpbubblefactory/gui.py
==========================
Tkinter monitor for the ExoticMinimizer.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time

class OptimizerDashboard:
    def __init__(self, title="Warp Bubble Optimizer Dashboard"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("600x450")
        self.root.configure(bg="#2d2d2d")
        
        # Abort Event Flag
        self.abort_event = threading.Event()
        
        self._setup_ui()
        
    def _setup_ui(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#2d2d2d')
        style.configure('TLabel', background='#2d2d2d', foreground='#ffffff', font=('Consolas', 11))
        style.configure('TButton', font=('Consolas', 11, 'bold'))
        
        main_frame = ttk.Frame(self.root, padding="15 15 15 15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Labels
        self.status_var = tk.StringVar(value="Status: WAITING")
        ttk.Label(main_frame, textvariable=self.status_var, font=('Consolas', 12, 'bold')).pack(anchor=tk.W, pady=(0, 10))
        
        self.best_exotic_var = tk.StringVar(value="Best Exotic Matter: N/A")
        ttk.Label(main_frame, textvariable=self.best_exotic_var, foreground="#4caf50").pack(anchor=tk.W)
        
        self.best_params_var = tk.StringVar(value="Best Params: N/A")
        ttk.Label(main_frame, textvariable=self.best_params_var).pack(anchor=tk.W, pady=(0, 15))
        
        # Log Text Area
        self.log_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, width=60, height=15, 
                                                  bg="#1e1e1e", fg="#00ff00", font=('Consolas', 10))
        self.log_area.pack(fill=tk.BOTH, expand=True)
        
        # Button Frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Stop Button
        self.stop_btn = tk.Button(btn_frame, text="🛑 ABORT OPTIMIZATION", bg="#d32f2f", fg="white", 
                                  font=('Consolas', 11, 'bold'), command=self._abort_clicked)
        self.stop_btn.pack(side=tk.RIGHT, ipadx=10, ipady=5)
        
        self.close_btn = tk.Button(btn_frame, text="Close", bg="#555555", fg="white",
                                   font=('Consolas', 11), command=self.root.destroy, state=tk.DISABLED)
        self.close_btn.pack(side=tk.LEFT, ipadx=10, ipady=5)

    def _abort_clicked(self):
        self.abort_event.set()
        self.update_status("Status: ABORTING (Waiting for current eval to finish...)")
        self.stop_btn.config(state=tk.DISABLED, bg="#772222")

    def update_status(self, text):
        self._safe_update(lambda: self.status_var.set(text))
        
    def update_best(self, exotic_val, params):
        self._safe_update(lambda: self.best_exotic_var.set(f"Best Exotic Matter: {exotic_val:.3e} J"))
        self._safe_update(lambda: self.best_params_var.set(f"Best Params: {params}"))
        
    def add_log(self, msg):
        def _add():
            self.log_area.insert(tk.END, msg + "\n")
            self.log_area.see(tk.END)
        self._safe_update(_add)
        
    def on_finish(self, msg):
        self.add_log("\n--- OPTIMIZATION FINISHED ---")
        self.add_log(msg)
        self.update_status("Status: FINISHED")
        self._safe_update(lambda: self.stop_btn.config(state=tk.DISABLED))
        self._safe_update(lambda: self.close_btn.config(state=tk.NORMAL, bg="#2196F3"))
        
    def _safe_update(self, func):
        try:
            self.root.after(0, func)
        except Exception:
            pass

    def check_abort(self):
        return self.abort_event.is_set()

    def run(self):
        self.root.mainloop()
