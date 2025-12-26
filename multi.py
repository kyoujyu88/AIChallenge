import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox, ttk  # ttkは新しいデザイン部品です
import threading
from llama_cpp import Llama
import pandas as pd
import sys
import glob
import os


class AIChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("渚のAI分析アシスタント (マルチモデル版)")
        self.root.geometry("800x800")

        # --- 1. デザイン部分 ---
        # ★最上部：モデル選択エリア（新設！）
        model_frame = tk.Frame(root, bg="#e0e0e0", pady=5)
        model_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(model_frame, text="使用モデル:", bg="#e0e0e0", font=("Meiryo", 10)).pack(side=tk.LEFT, padx=10)

        # モデル一覧を取得してコンボボックス（プルダウン）を作ります
        self.model_files = glob.glob("gguf/*.gguf")  # ggufフォルダの中身を検索
        if not self.model_files:
            self.model_files = ["モデルが見つかりません"]

        self.model_combo = ttk.Combobox(model_frame, values=self.model_files, width=50, state="readonly")
        # 最初に見つかったモデルをデフォルトで選択
        if self.model_files:
            try:
                self.model_combo.current(0)
            except Exception:
                pass
        self.model_combo.pack(side=tk.LEFT, padx=5)

        # 切り替えボタン
        self.change_btn = tk.Button(
            model_frame,
            text="切替・再読込",
            command=self.reload_model_trigger,
            bg="#98fb98",
            font=("Meiryo", 9),
        )
        self.change_btn.pack(side=tk.LEFT, padx=5)

        # ★操作エリア
        top_frame = tk.Frame(root, bg="#f0f0f0", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # CSVボタン
        self.csv_btn = tk.Button(
            top_frame,
            text="📂 CSV読込",
            command=self.load_csv,
            bg="#87ceeb",
            fg="white",
            font=("Meiryo", 10, "bold"),
            width=12,
        )
        self.csv_btn.pack(side=tk.LEFT, padx=10)

        # 入力欄
        self.input_entry = tk.Entry(top_frame, font=("Meiryo", 12))
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.input_entry.bind("<Return>", self.send_message)

        # 送信ボタン
        self.send_btn = tk.Button(
            top_frame,
            text="送信",
            command=self.send_message,
            bg="#ffb6c1",
            fg="white",
            font=("Meiryo", 10, "bold"),
            width=10,
        )
        self.send_btn.pack(side=tk.RIGHT, padx=10)

        # ★ログエリア
        self.log_area = scrolledtext.ScrolledText(root, font=("Meiryo", 11), state="disabled")
        self.log_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # --- 2. AIの準備 ---
        self.system_prompt = "システム: あなたはデータ分析が得意なAIアシスタントです。論理的に回答してください。\n"
        self.history = self.system_prompt
        self.llm = None

        # 起動時に最初のモデルを読み込みます
        self.reload_model_trigger()

    def reload_model_trigger(self):
        # ボタンが押されたらモデル読み込みスレッドを開始
        selected_model = self.model_combo.get()
        if selected_model == "モデルが見つかりません":
            messagebox.showerror("エラー", "ggufフォルダにモデルファイルを入れてください！")
            return

        self.append_log("システム", f"「{os.path.basename(selected_model)}」に切り替えています...（お待ちください）")
        self.root.title("渚のAI分析アシスタント (モデル読込中..)")
        self.change_btn.config(state="disabled")  # 連打防止
        self.send_btn.config(state="disabled")

        threading.Thread(target=self.load_model, args=(selected_model,), daemon=True).start()

    def load_model(self, model_path):
        try:
            # 既存のモデルがあればメモリを解放...できればいいですがPython任せにします
            self.llm = None

            # モデル読み込み
            self.llm = Llama(
                model_path=model_path,
                n_ctx=8192,  # 記憶力MAX
                n_threads=4,
                n_batch=512,
                verbose=False,
            )

            # 完了報告
            self.root.after(0, self.post_load_success, model_path)

        except Exception as e:
            self.root.after(0, self.append_log, "エラー", f"モデル読込失敗: {e}")
            self.root.after(0, lambda: self.change_btn.config(state="normal"))

    def post_load_success(self, model_path):
        # 読み込み完了後の画面更新
        model_name = os.path.basename(model_path)
        self.append_log("システム", f"準備完了！今は「{model_name}」が担当します。")
        self.root.title(f"渚のAI分析アシスタント - {model_name}")
        self.change_btn.config(state="normal")
        self.send_btn.config(state="normal")

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSVファイル", "*.csv")])
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)

            info_text = f"【データ概要】\n行数: {df.shape[0]}, 列数: {df.shape[1]}\n\n"
            info_text += f"【列名一覧】\n{', '.join(df.columns)}\n\n"
            info_text += f"【データの先頭5行】\n{df.head().to_string()}\n\n"
            info_text += f"【統計情報】\n{df.describe().to_string()}"

            self.append_log("システム", f"「{os.path.basename(file_path)}」を読み込みました！")

            data_prompt = f"ユーザー: 以下のCSVデータを読み込みました。\n{info_text}\nシステム: データを読み込みました。\n"
            self.history += data_prompt

        except Exception as e:
            messagebox.showerror("エラー", f"CSVエラー: {e}")

    def send_message(self, event=None):
        user_text = self.input_entry.get()
        if not user_text or self.llm is None:
            return

        self.append_log("あなた", user_text)
        self.input_entry.delete(0, tk.END)
        self.history += f"ユーザー: {user_text}\nシステム:"

        self.root.title("考え中...")
        threading.Thread(target=self.run_generation, daemon=True).start()

    def run_generation(self):
        try:
            output = self.llm(
                self.history,
                max_tokens=500,
                temperature=0.1,  # 論理的モード
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["ユーザー:", "\n\n"],
                echo=False,
            )
            response = output["choices"][0]["text"].strip()

            self.history += f" {response}\n"

            self.root.after(0, self.append_log, "AI", response)
            # タイトルをモデル名に戻す
            model_name = os.path.basename(self.model_combo.get())
            self.root.after(0, lambda: self.root.title(f"渚のAI分析アシスタント - {model_name}"))

        except Exception as e:
            self.root.after(0, self.append_log, "エラー", str(e))

    def append_log(self, sender, text):
        self.log_area.config(state="normal")
        self.log_area.insert(tk.END, f"[{sender}] {text}\n\n")
        self.log_area.see(tk.END)
        self.log_area.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    app = AIChatApp(root)
    root.mainloop()
