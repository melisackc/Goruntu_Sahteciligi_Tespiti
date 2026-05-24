import tkinter as tk
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk
from datetime import datetime

from core.orb_detector import ORBDetector
from core.akaze_detector import AKAZEDetector
from core.sift_detector import SIFTDetector
from core.surf_detector import SURFDetector
from core.cnn_detector import CNNDetector
from core.lstm_detector import LSTMDetector

# Renk tanımlamaları
COLOR_BG = "#0c0c0e"
COLOR_CARD = "#16161a"
COLOR_TEXT_MAIN = "#ffffff"
COLOR_TEXT_BODY = "#e1e1e6"
COLOR_MUTED = "#8f8f98"
COLOR_BORDER = "#242429"
COLOR_CONSOLE_BG = "#08080a"

COLOR_SELECT = "#5c3dcc"
COLOR_SELECT_HOVER = "#4b2fb3"

COLOR_CLASSIC = "#22222a"
COLOR_CLASSIC_HOVER = "#32323e"
COLOR_CLASSIC_FG = "#e1e1e6"

COLOR_AI = "#007acc"
COLOR_AI_HOVER = "#005999"

COLOR_REPORT = "#04d361"
COLOR_REPORT_HOVER = "#02a247"

COLOR_RESET = "#3e3e47"
COLOR_RESET_HOVER = "#2e2e36"

COLOR_BATCH = "#ff8c00"
COLOR_BATCH_HOVER = "#e07b00"

COLOR_SUCCESS = "#04d361"
COLOR_WARNING = "#ffcc00"
COLOR_ERROR = "#ff3333"

class MainWindow:
    # Arayüz sınıfı

    def __init__(self):
        try:
            self.root = tk.Tk()
            self.root.title("Görüntü Sahteciliği Tespit Sistemi")
            self.root.state("zoomed")
            self.is_fullscreen = False

            # Tam ekran kısayolları
            self.root.bind("<F11>", self.toggle_fullscreen)
            self.root.bind("<Escape>", self.exit_fullscreen)

            self.selected_image_path = None
            self.image_preview = None
            self.last_analysis_result = None

            self.create_widgets()

            # Pencere boyutu değiştikçe resmi sığdır
            self.left_frame.bind("<Configure>", self.on_resize_left_frame)
        except Exception as e:
            messagebox.showerror("Başlatma Hatası", f"Arayüz yüklenirken kritik bir hata oluştu:\n{e}")

    def create_widgets(self):
        try:
            self.root.configure(bg=COLOR_BG)

            # Üst Header
            header_frame = tk.Frame(self.root, bg=COLOR_BG)
            header_frame.pack(fill="x", pady=(20, 10))

            title_label = tk.Label(
                header_frame,
                text="Görüntü Sahteciliği Tespit Sistemi",
                font=("Segoe UI", 24, "bold"),
                bg=COLOR_BG,
                fg=COLOR_TEXT_MAIN
            )
            title_label.pack()

            subtitle_label = tk.Label(
                header_frame,
                text="ORB, AKAZE, SIFT, SURF ve AI Tabanlı Analiz Modülleri Dashboard",
                font=("Segoe UI", 10),
                bg=COLOR_BG,
                fg=COLOR_MUTED
            )
            subtitle_label.pack(pady=(3, 0))

            # Ana yerleşim çerçevesi
            main_frame = tk.Frame(self.root, bg=COLOR_BG)
            main_frame.pack(fill="both", expand=True, padx=30, pady=(10, 25))
            
            main_frame.rowconfigure(0, weight=1)
            main_frame.columnconfigure(0, weight=3)
            main_frame.columnconfigure(1, weight=2)

            # Sol panel (Görsel önizleme)
            self.left_frame = tk.Frame(
                main_frame,
                bg=COLOR_CARD,
                highlightthickness=1,
                highlightbackground=COLOR_BORDER
            )
            self.left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
            self.left_frame.pack_propagate(False)

            self.image_label = tk.Label(
                self.left_frame,
                text="Henüz görüntü seçilmedi\n\n(Desteklenen formatlar: JPG, JPEG, PNG, GIF, BMP)",
                font=("Segoe UI", 12),
                bg=COLOR_CARD,
                fg=COLOR_MUTED,
                justify="center"
            )
            self.image_label.pack(expand=True, fill="both", padx=20, pady=20)

            # Sağ panel (Kontroller ve konsol)
            right_frame = tk.Frame(main_frame, bg=COLOR_BG)
            right_frame.grid(row=0, column=1, sticky="nsew", padx=(15, 0))
            
            right_frame.columnconfigure(0, weight=1)
            right_frame.rowconfigure(0, weight=0)
            right_frame.rowconfigure(1, weight=1)

            # Kontrol paneli
            control_card = tk.Frame(
                right_frame,
                bg=COLOR_CARD,
                highlightthickness=1,
                highlightbackground=COLOR_BORDER,
                padx=20,
                pady=15
            )
            control_card.grid(row=0, column=0, sticky="ew", pady=(0, 15))
            
            control_title = tk.Label(
                control_card,
                text="Kontrol Paneli",
                font=("Segoe UI", 14, "bold"),
                bg=COLOR_CARD,
                fg=COLOR_TEXT_MAIN
            )
            control_title.pack(anchor="w", pady=(0, 10))

            btn_grid_frame = tk.Frame(control_card, bg=COLOR_CARD)
            btn_grid_frame.pack(fill="x", expand=True)
            btn_grid_frame.columnconfigure(0, weight=1)
            btn_grid_frame.columnconfigure(1, weight=1)

            # Görüntü seçme butonu
            self.select_btn = tk.Button(
                btn_grid_frame,
                text="📂 Görüntü Seç",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_SELECT,
                fg="#ffffff",
                activebackground=COLOR_SELECT_HOVER,
                activeforeground="#ffffff",
                bd=0,
                height=2,
                cursor="hand2",
                command=self.select_image
            )
            self.select_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

            # Klasik algoritmalar
            self.orb_btn = tk.Button(
                btn_grid_frame,
                text="🔍 ORB ile Analiz Et",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_CLASSIC,
                fg=COLOR_CLASSIC_FG,
                activebackground=COLOR_CLASSIC_HOVER,
                activeforeground=COLOR_CLASSIC_FG,
                bd=0,
                height=2,
                cursor="hand2",
                command=self.orb_analysis
            )
            self.orb_btn.grid(row=1, column=0, sticky="ew", padx=(0, 5), pady=(0, 8))

            self.akaze_btn = tk.Button(
                btn_grid_frame,
                text="⚡ AKAZE ile Analiz Et",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_CLASSIC,
                fg=COLOR_CLASSIC_FG,
                activebackground=COLOR_CLASSIC_HOVER,
                activeforeground=COLOR_CLASSIC_FG,
                bd=0,
                height=2,
                cursor="hand2",
                command=self.akaze_analysis
            )
            self.akaze_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0), pady=(0, 8))

            self.sift_btn = tk.Button(
                btn_grid_frame,
                text="🧬 SIFT ile Analiz Et",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_CLASSIC,
                fg=COLOR_CLASSIC_FG,
                activebackground=COLOR_CLASSIC_HOVER,
                activeforeground=COLOR_CLASSIC_FG,
                bd=0,
                height=2,
                cursor="hand2",
                command=self.sift_analysis
            )
            self.sift_btn.grid(row=2, column=0, sticky="ew", padx=(0, 5), pady=(0, 8))

            self.surf_btn = tk.Button(
                btn_grid_frame,
                text="🌊 SURF ile Analiz Et",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_CLASSIC,
                fg=COLOR_CLASSIC_FG,
                activebackground=COLOR_CLASSIC_HOVER,
                activeforeground=COLOR_CLASSIC_FG,
                bd=0,
                height=2,
                cursor="hand2",
                command=self.surf_analysis
            )
            self.surf_btn.grid(row=2, column=1, sticky="ew", padx=(5, 0), pady=(0, 8))

            # Yapay zeka modülleri
            self.cnn_btn = tk.Button(
                btn_grid_frame,
                text="🧠 AI CNN ile Analiz",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_AI,
                fg="#ffffff",
                activebackground=COLOR_AI_HOVER,
                activeforeground="#ffffff",
                bd=0,
                height=2,
                cursor="hand2",
                command=self.cnn_analysis
            )
            self.cnn_btn.grid(row=3, column=0, sticky="ew", padx=(0, 5), pady=(0, 8))

            self.lstm_btn = tk.Button(
                btn_grid_frame,
                text="⛓️ AI LSTM ile Analiz",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_AI,
                fg="#ffffff",
                activebackground=COLOR_AI_HOVER,
                activeforeground="#ffffff",
                bd=0,
                height=2,
                cursor="hand2",
                command=self.lstm_analysis
            )
            self.lstm_btn.grid(row=3, column=1, sticky="ew", padx=(5, 0), pady=(0, 8))

            # Raporlama ve sıfırlama
            self.report_btn = tk.Button(
                btn_grid_frame,
                text="📋 Rapor Oluştur",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_REPORT,
                fg="#ffffff",
                activebackground=COLOR_REPORT_HOVER,
                activeforeground="#ffffff",
                bd=0,
                height=2,
                cursor="hand2",
                command=self.create_report
            )
            self.report_btn.grid(row=4, column=0, sticky="ew", padx=(0, 5), pady=(0, 5))

            self.reset_btn = tk.Button(
                btn_grid_frame,
                text="🔄 Sonucu Sıfırla",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_RESET,
                fg="#ffffff",
                activebackground=COLOR_RESET_HOVER,
                activeforeground="#ffffff",
                bd=0,
                height=2,
                cursor="hand2",
                command=self.reset_results
            )
            self.reset_btn.grid(row=4, column=1, sticky="ew", padx=(5, 0), pady=(0, 5))

            # Toplu test butonu
            self.batch_btn = tk.Button(
                btn_grid_frame,
                text="🚀 Tüm Algoritmaları Test Et",
                font=("Segoe UI", 11, "bold"),
                bg=COLOR_BATCH,
                fg="#ffffff",
                activebackground=COLOR_BATCH_HOVER,
                activeforeground="#ffffff",
                bd=0,
                height=2,
                cursor="hand2",
                command=self.batch_analysis
            )
            self.batch_btn.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))

            self.setup_hover_effects()

            # Analiz Konsolu
            console_card = tk.Frame(
                right_frame,
                bg=COLOR_CARD,
                highlightthickness=1,
                highlightbackground=COLOR_BORDER,
                padx=20,
                pady=15
            )
            console_card.grid(row=1, column=0, sticky="nsew")
            console_card.columnconfigure(0, weight=1)
            console_card.rowconfigure(1, weight=1)

            console_title = tk.Label(
                console_card,
                text="Analiz Konsolu / Bilgi",
                font=("Segoe UI", 13, "bold"),
                bg=COLOR_CARD,
                fg=COLOR_TEXT_MAIN
            )
            console_title.grid(row=0, column=0, sticky="w", pady=(0, 8))

            console_box = tk.Frame(
                console_card,
                bg=COLOR_CONSOLE_BG,
                highlightthickness=1,
                highlightbackground=COLOR_BORDER
            )
            console_box.grid(row=1, column=0, sticky="nsew")

            self.scrollbar = tk.Scrollbar(console_box, bg=COLOR_CONSOLE_BG, highlightthickness=0)
            self.scrollbar.pack(side="right", fill="y")

            self.result_text = tk.Text(
                console_box,
                font=("Consolas", 10),
                bg=COLOR_CONSOLE_BG,
                fg=COLOR_MUTED,
                bd=0,
                padx=12,
                pady=12,
                wrap="word",
                yscrollcommand=self.scrollbar.set
            )
            self.result_text.pack(side="left", fill="both", expand=True)
            self.scrollbar.config(command=self.result_text.yview)

            self.write_to_console("[DURUM] Sistem hazır. Lütfen bir görsel seçin.", COLOR_MUTED)

        except Exception as e:
            messagebox.showerror("Bileşen Hatası", f"Arayüz yerleşimi yapılırken hata:\n{e}")

    def write_to_console(self, text, color=COLOR_TEXT_BODY):
        try:
            self.result_text.config(state="normal")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", text)
            self.result_text.config(fg=color, state="disabled")
            self.result_text.yview_moveto(0)
        except Exception as e:
            print(f"[KONSOL HATASI] Yazı konsola eklenirken hata: {e}")

    def setup_hover_effects(self):
        def bind_hover(button, bg_normal, bg_hover):
            button.bind("<Enter>", lambda e: button.config(bg=bg_hover))
            button.bind("<Leave>", lambda e: button.config(bg=bg_normal))

        bind_hover(self.select_btn, COLOR_SELECT, COLOR_SELECT_HOVER)
        bind_hover(self.orb_btn, COLOR_CLASSIC, COLOR_CLASSIC_HOVER)
        bind_hover(self.akaze_btn, COLOR_CLASSIC, COLOR_CLASSIC_HOVER)
        bind_hover(self.sift_btn, COLOR_CLASSIC, COLOR_CLASSIC_HOVER)
        bind_hover(self.surf_btn, COLOR_CLASSIC, COLOR_CLASSIC_HOVER)
        bind_hover(self.cnn_btn, COLOR_AI, COLOR_AI_HOVER)
        bind_hover(self.lstm_btn, COLOR_AI, COLOR_AI_HOVER)
        bind_hover(self.report_btn, COLOR_REPORT, COLOR_REPORT_HOVER)
        bind_hover(self.reset_btn, COLOR_RESET, COLOR_RESET_HOVER)
        bind_hover(self.batch_btn, COLOR_BATCH, COLOR_BATCH_HOVER)

    def select_image(self):
        try:
            file_path = filedialog.askopenfilename(
                title="Analiz Edilecek Görseli Seçin",
                filetypes=[
                    ("Görsel Dosyaları", "*.jpg *.jpeg *.png *.gif *.bmp"),
                    ("Tüm Dosyalar", "*.*")
                ]
            )
            
            if file_path:
                self.selected_image_path = file_path
                self.show_image(file_path)
                self.last_analysis_result = None
                
                file_name = os.path.basename(file_path)
                file_size_kb = os.path.getsize(file_path) / 1024
                
                console_msg = (
                    f"[BİLGİ] Görüntü başarıyla yüklendi.\n"
                    f"-----------------------------------\n"
                    f"Dosya Adı: {file_name}\n"
                    f"Boyut: {file_size_kb:.2f} KB\n"
                    f"Dosya Yolu: {file_path}\n\n"
                    f"[DURUM] Analiz için hazır. Klasik veya AI tabanlı analizleri başlatabilirsiniz."
                )
                self.write_to_console(console_msg, COLOR_SUCCESS)
        except Exception as e:
            messagebox.showerror("Hata", f"Görüntü seçilemedi:\n{e}")
            self.write_to_console(f"[HATA] Dosya seçme başarısız.\n{e}", COLOR_ERROR)

    def show_image(self, file_path):
        try:
            frame_w = self.left_frame.winfo_width()
            frame_h = self.left_frame.winfo_height()

            if frame_w < 50:
                frame_w = 460
            if frame_h < 50:
                frame_h = 550

            max_w = max(frame_w - 40, 100)
            max_h = max(frame_h - 40, 100)

            img = Image.open(file_path)
            img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
            
            self.image_preview = ImageTk.PhotoImage(img)
            self.image_label.config(image=self.image_preview, text="")
            self.image_label.image = self.image_preview
        except Exception as e:
            messagebox.showerror("Önizleme Hatası", f"Görsel önizlenemedi:\n{e}")
            self.image_label.config(image="", text="Görsel önizlemesi yüklenemedi.", fg=COLOR_ERROR)

    def on_resize_left_frame(self, event):
        if self.selected_image_path:
            self.show_image(self.selected_image_path)

    def format_analysis_result(self, result):
        try:
            algo_raw = result.get("algorithm", "")
            if "ORB" in algo_raw:
                algo_name = "ORB"
            elif "AKAZE" in algo_raw:
                algo_name = "AKAZE"
            elif "SIFT" in algo_raw:
                algo_name = "SIFT"
            elif "SURF" in algo_raw:
                algo_name = "SURF"
            elif "CNN" in algo_raw:
                algo_name = "CNN"
            elif "LSTM" in algo_raw:
                algo_name = "LSTM"
            else:
                algo_name = algo_raw

            file_path = self.selected_image_path if self.selected_image_path else "Seçilmedi"
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            risk_level = result.get("risk_level", "Düşük Risk")
            risk_score = result.get("risk_score", 0)

            if risk_level == "Desteklenmiyor":
                report = (
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                    "ANALİZ RAPORU\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                    f"Algoritma: {algo_name}\n"
                    f"Seçilen Dosya: {file_path}\n"
                    f"Analiz Tarihi: {date_str}\n\n"
                    "Teknik Bulgular:\n"
                    "- Tespit Edilen Anahtar Nokta Sayısı: 0\n"
                    "- Şüpheli Eşleşme Sayısı: 0\n"
                    "- Risk Skoru: 0 / 100\n"
                    f"- Risk Seviyesi: {risk_level}\n\n"
                    "Yorum:\n"
                    f"{result.get('message', '')}\n\n"
                    "Öneri:\n"
                    f"- {result.get('recommendation', '')}"
                )
                return report

            if algo_name == "CNN":
                tech_findings = (
                    f"- Özellik Skoru: %{result.get('keypoint_count', 0)}\n"
                    f"- Anomali Skoru: %{result.get('match_count', 0)}"
                )
            elif algo_name == "LSTM":
                tech_findings = (
                    f"- Özellik Skoru: {result.get('keypoint_count', 0)}\n"
                    f"- Anomali Skoru: {result.get('match_count', 0)}"
                )
            else:
                tech_findings = (
                    f"- Tespit Edilen Anahtar Nokta Sayısı: {result.get('keypoint_count', 0)}\n"
                    f"- Şüpheli Eşleşme Sayısı: {result.get('match_count', 0)}"
                )

            report = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "ANALİZ RAPORU\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Algoritma: {algo_name}\n"
                f"Seçilen Dosya: {file_path}\n"
                f"Analiz Tarihi: {date_str}\n\n"
                "Teknik Bulgular:\n"
                f"{tech_findings}\n"
                f"- Risk Skoru: {risk_score} / 100\n"
                f"- Risk Seviyesi: {risk_level}\n\n"
                "Yorum:\n"
                f"{result.get('message', '')}\n\n"
                "Öneri:\n"
                f"- {result.get('recommendation', '')}"
            )
            return report
        except Exception as e:
            return f"Rapor formatlanırken hata oluştu: {str(e)}"

    def orb_analysis(self):
        try:
            if not self.selected_image_path:
                messagebox.showwarning("Uyarı", "Lütfen önce analiz edilecek bir görüntü seçin!")
                self.write_to_console("[UYARI] Görsel seçilmediği için işlem başlatılamadı.", COLOR_WARNING)
                return

            self.write_to_console("[DURUM] ORB analizi yapılıyor, lütfen bekleyin...", COLOR_TEXT_BODY)
            self.root.update_idletasks()

            detector = ORBDetector()
            result = detector.analyze(self.selected_image_path)

            if result["risk_level"] == "Düşük Risk":
                color = COLOR_SUCCESS
            elif result["risk_level"] == "Orta Risk":
                color = COLOR_WARNING
            else:
                color = COLOR_ERROR

            result_msg = self.format_analysis_result(result)
            self.last_analysis_result = result_msg
            self.write_to_console(result_msg, color)

        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"ORB analizi gerçekleştirilemedi:\n{e}")
            self.write_to_console(f"[HATA] ORB analiz hatası.\nDetay: {e}", COLOR_ERROR)

    def akaze_analysis(self):
        try:
            if not self.selected_image_path:
                messagebox.showwarning("Uyarı", "Lütfen önce analiz edilecek bir görüntü seçin!")
                self.write_to_console("[UYARI] Görsel seçilmediği için işlem başlatılamadı.", COLOR_WARNING)
                return

            self.write_to_console("[DURUM] AKAZE analizi yapılıyor, lütfen bekleyin...", COLOR_TEXT_BODY)
            self.root.update_idletasks()

            detector = AKAZEDetector()
            result = detector.analyze(self.selected_image_path)

            if result["risk_level"] == "Düşük Risk":
                color = COLOR_SUCCESS
            elif result["risk_level"] == "Orta Risk":
                color = COLOR_WARNING
            else:
                color = COLOR_ERROR

            result_msg = self.format_analysis_result(result)
            self.last_analysis_result = result_msg
            self.write_to_console(result_msg, color)

        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"AKAZE analizi gerçekleştirilemedi:\n{e}")
            self.write_to_console(f"[HATA] AKAZE analiz hatası.\nDetay: {e}", COLOR_ERROR)

    def sift_analysis(self):
        try:
            if not self.selected_image_path:
                messagebox.showwarning("Uyarı", "Lütfen önce analiz edilecek bir görüntü seçin!")
                self.write_to_console("[UYARI] Görsel seçilmediği için işlem başlatılamadı.", COLOR_WARNING)
                return

            self.write_to_console("[DURUM] SIFT analizi yapılıyor, lütfen bekleyin...", COLOR_TEXT_BODY)
            self.root.update_idletasks()

            detector = SIFTDetector()
            result = detector.analyze(self.selected_image_path)

            if result["risk_level"] == "Düşük Risk":
                color = COLOR_SUCCESS
            elif result["risk_level"] == "Orta Risk":
                color = COLOR_WARNING
            else:
                color = COLOR_ERROR

            result_msg = self.format_analysis_result(result)
            self.last_analysis_result = result_msg
            self.write_to_console(result_msg, color)

        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"SIFT analizi gerçekleştirilemedi:\n{e}")
            self.write_to_console(f"[HATA] SIFT analiz hatası.\nDetay: {e}", COLOR_ERROR)

    def surf_analysis(self):
        try:
            if not self.selected_image_path:
                messagebox.showwarning("Uyarı", "Lütfen önce analiz edilecek bir görüntü seçin!")
                self.write_to_console("[UYARI] Görsel seçilmediği için işlem başlatılamadı.", COLOR_WARNING)
                return

            self.write_to_console("[DURUM] SURF analizi yapılıyor, lütfen bekleyin...", COLOR_TEXT_BODY)
            self.root.update_idletasks()

            detector = SURFDetector()
            result = detector.analyze(self.selected_image_path)

            if result["risk_level"] == "Desteklenmiyor":
                color = COLOR_WARNING
            else:
                if result["risk_level"] == "Düşük Risk":
                    color = COLOR_SUCCESS
                elif result["risk_level"] == "Orta Risk":
                    color = COLOR_WARNING
                else:
                    color = COLOR_ERROR

            result_msg = self.format_analysis_result(result)
            self.last_analysis_result = result_msg
            self.write_to_console(result_msg, color)

        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"SURF analizi gerçekleştirilemedi:\n{e}")
            self.write_to_console(f"[HATA] SURF analiz hatası.\nDetay: {e}", COLOR_ERROR)

    def cnn_analysis(self):
        try:
            if not self.selected_image_path:
                messagebox.showwarning("Uyarı", "Lütfen önce analiz edilecek bir görüntü seçin!")
                self.write_to_console("[UYARI] Görsel seçilmediği için işlem başlatılamadı.", COLOR_WARNING)
                return

            self.write_to_console("[DURUM] AI CNN analizi yapılıyor, lütfen bekleyin...", COLOR_TEXT_BODY)
            self.root.update_idletasks()

            detector = CNNDetector()
            result = detector.analyze(self.selected_image_path)

            if result["risk_level"] == "Düşük Risk":
                color = COLOR_SUCCESS
            elif result["risk_level"] == "Orta Risk":
                color = COLOR_WARNING
            else:
                color = COLOR_ERROR

            result_msg = self.format_analysis_result(result)
            self.last_analysis_result = result_msg
            self.write_to_console(result_msg, color)

        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"CNN AI analizi gerçekleştirilemedi:\n{e}")
            self.write_to_console(f"[HATA] CNN AI analiz hatası.\nDetay: {e}", COLOR_ERROR)

    def lstm_analysis(self):
        try:
            if not self.selected_image_path:
                messagebox.showwarning("Uyarı", "Lütfen önce analiz edilecek bir görüntü seçin!")
                self.write_to_console("[UYARI] Görsel seçilmediği için işlem başlatılamadı.", COLOR_WARNING)
                return

            self.write_to_console("[DURUM] AI LSTM analizi yapılıyor, lütfen bekleyin...", COLOR_TEXT_BODY)
            self.root.update_idletasks()

            detector = LSTMDetector()
            result = detector.analyze(self.selected_image_path)

            if result["risk_level"] == "Düşük Risk":
                color = COLOR_SUCCESS
            elif result["risk_level"] == "Orta Risk":
                color = COLOR_WARNING
            else:
                color = COLOR_ERROR

            result_msg = self.format_analysis_result(result)
            self.last_analysis_result = result_msg
            self.write_to_console(result_msg, color)

        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"LSTM AI analizi gerçekleştirilemedi:\n{e}")
            self.write_to_console(f"[HATA] LSTM AI analiz hatası.\nDetay: {e}", COLOR_ERROR)

    def batch_analysis(self):
        try:
            if not self.selected_image_path:
                messagebox.showwarning("Uyarı", "Lütfen önce bir görüntü seçin.")
                self.write_to_console("[UYARI] Görsel seçilmediği için işlem başlatılamadı.", COLOR_WARNING)
                return

            self.write_to_console("[DURUM] Tüm algoritmalar sırayla test ediliyor, lütfen bekleyin...\n\n"
                                  "ORB çalıştırılıyor...\n"
                                  "AKAZE çalıştırılıyor...\n"
                                  "SIFT çalıştırılıyor...\n"
                                  "SURF çalıştırılıyor...\n"
                                  "CNN çalıştırılıyor...\n"
                                  "LSTM çalıştırılıyor...", COLOR_TEXT_BODY)
            self.root.update_idletasks()

            results = []

            # 1. ORB
            try:
                detector = ORBDetector()
                orb_res = detector.analyze(self.selected_image_path)
            except Exception as e:
                orb_res = {
                    "algorithm": "ORB Algoritması",
                    "keypoint_count": 0,
                    "match_count": 0,
                    "risk_score": None,
                    "risk_level": "Hata",
                    "message": f"Bu algoritmada hata oluştu: {str(e)}",
                    "recommendation": "Hata oluştu."
                }
            results.append(orb_res)

            # 2. AKAZE
            try:
                detector = AKAZEDetector()
                akaze_res = detector.analyze(self.selected_image_path)
            except Exception as e:
                akaze_res = {
                    "algorithm": "AKAZE Algoritması",
                    "keypoint_count": 0,
                    "match_count": 0,
                    "risk_score": None,
                    "risk_level": "Hata",
                    "message": f"Bu algoritmada hata oluştu: {str(e)}",
                    "recommendation": "Hata oluştu."
                }
            results.append(akaze_res)

            # 3. SIFT
            try:
                detector = SIFTDetector()
                sift_res = detector.analyze(self.selected_image_path)
            except Exception as e:
                sift_res = {
                    "algorithm": "SIFT Algoritması",
                    "keypoint_count": 0,
                    "match_count": 0,
                    "risk_score": None,
                    "risk_level": "Hata",
                    "message": f"Bu algoritmada hata oluştu: {str(e)}",
                    "recommendation": "Hata oluştu."
                }
            results.append(sift_res)

            # 4. SURF
            try:
                detector = SURFDetector()
                surf_res = detector.analyze(self.selected_image_path)
            except Exception as e:
                surf_res = {
                    "algorithm": "SURF Algoritması",
                    "keypoint_count": 0,
                    "match_count": 0,
                    "risk_score": None,
                    "risk_level": "Hata",
                    "message": f"Bu algoritmada hata oluştu: {str(e)}",
                    "recommendation": "Hata oluştu."
                }
            results.append(surf_res)

            # 5. CNN
            try:
                detector = CNNDetector()
                cnn_res = detector.analyze(self.selected_image_path)
            except Exception as e:
                cnn_res = {
                    "algorithm": "CNN Tabanlı AI Analizi",
                    "keypoint_count": 0,
                    "match_count": 0,
                    "risk_score": None,
                    "risk_level": "Hata",
                    "message": f"Bu algoritmada hata oluştu: {str(e)}",
                    "recommendation": "Hata oluştu."
                }
            results.append(cnn_res)

            # 6. LSTM
            try:
                detector = LSTMDetector()
                lstm_res = detector.analyze(self.selected_image_path)
            except Exception as e:
                lstm_res = {
                    "algorithm": "LSTM Tabanlı Prototip AI Analizi",
                    "keypoint_count": 0,
                    "match_count": 0,
                    "risk_score": None,
                    "risk_level": "Hata",
                    "message": f"Bu algoritmada hata oluştu: {str(e)}",
                    "recommendation": "Hata oluştu."
                }
            results.append(lstm_res)

            def get_block_text(res):
                rl = res.get("risk_level", "Düşük Risk")
                if rl == "Desteklenmiyor":
                    return (
                        f"- Risk Seviyesi: Desteklenmiyor\n"
                        f"- Açıklama: {res.get('message', '')}"
                    )
                elif rl == "Hata":
                    return (
                        f"- Hata: {res.get('message', '')}"
                    )

                algo_name = res.get("algorithm", "")
                if "CNN" in algo_name:
                    tech = f"- Özellik Skoru: %{res.get('keypoint_count', 0)}\n- Anomali Skoru: %{res.get('match_count', 0)}"
                elif "LSTM" in algo_name:
                    tech = f"- Özellik Skoru: {res.get('keypoint_count', 0)}\n- Anomali Skoru: {res.get('match_count', 0)}"
                else:
                    tech = f"- Tespit Edilen Anahtar Nokta Sayısı: {res.get('keypoint_count', 0)}\n- Şüpheli Eşleşme Sayısı: {res.get('match_count', 0)}"

                return (
                    f"{tech}\n"
                    f"- Risk Skoru: {res.get('risk_score', 0)} / 100\n"
                    f"- Risk Seviyesi: {rl}\n"
                    f"- Yorum: {res.get('message', '')}"
                )

            valid_results = []
            for res in results:
                if res.get("risk_level") not in ["Desteklenmiyor", "Hata"] and res.get("risk_score") is not None:
                    valid_results.append(res)

            if len(valid_results) > 0:
                max_res = max(valid_results, key=lambda x: x['risk_score'])
                min_res = min(valid_results, key=lambda x: x['risk_score'])
                avg_score = sum(x['risk_score'] for x in valid_results) / len(valid_results)

                def get_short_name(name):
                    if "ORB" in name: return "ORB"
                    if "AKAZE" in name: return "AKAZE"
                    if "SIFT" in name: return "SIFT"
                    if "SURF" in name: return "SURF"
                    if "CNN" in name: return "CNN"
                    if "LSTM" in name: return "LSTM"
                    return name

                max_name = f"{get_short_name(max_res['algorithm'])} (%{max_res['risk_score']})"
                min_name = f"{get_short_name(min_res['algorithm'])} (%{min_res['risk_score']})"
                avg_str = f"{avg_score:.1f} / 100"

                if avg_score <= 30:
                    general_comment = "Genel olarak düşük riskli görünmektedir."
                    color = COLOR_SUCCESS
                elif 31 <= avg_score <= 65:
                    general_comment = "Genel olarak orta riskli görünmektedir. Manuel inceleme önerilir."
                    color = COLOR_WARNING
                else:
                    general_comment = "Genel olarak yüksek riskli görünmektedir. Detaylı manuel inceleme önerilir."
            else:
                max_name = "Belirlenemedi"
                min_name = "Belirlenemedi"
                avg_str = "Hesaplanamadı"
                general_comment = "Analiz gerçekleştirilemedi."
                color = COLOR_ERROR

            summary_lines = []
            for res in results:
                algo_raw = res.get("algorithm", "")
                if "ORB" in algo_raw: short_name = "ORB"
                elif "AKAZE" in algo_raw: short_name = "AKAZE"
                elif "SIFT" in algo_raw: short_name = "SIFT"
                elif "SURF" in algo_raw: short_name = "SURF"
                elif "CNN" in algo_raw: short_name = "CNN"
                elif "LSTM" in algo_raw: short_name = "LSTM"
                else: short_name = algo_raw

                rl = res.get("risk_level", "Düşük Risk")
                if rl in ["Desteklenmiyor", "Hata"]:
                    summary_lines.append(f"- {short_name}: {rl}")
                else:
                    summary_lines.append(f"- {short_name}: {res.get('risk_score', 0)} / {rl}")
            summary_table = "\n".join(summary_lines)

            file_path = self.selected_image_path if self.selected_image_path else "Seçilmedi"
            date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            report_msg = (
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "TOPLU ANALİZ RAPORU\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"Seçilen Dosya: {file_path}\n"
                f"Analiz Tarihi: {date_str}\n\n"
                f"[1] ORB SONUCU\n"
                f"{get_block_text(results[0])}\n"
                f"--------------------------------\n\n"
                f"[2] AKAZE SONUCU\n"
                f"{get_block_text(results[1])}\n"
                f"--------------------------------\n\n"
                f"[3] SIFT SONUCU\n"
                f"{get_block_text(results[2])}\n"
                f"--------------------------------\n\n"
                f"[4] SURF SONUCU\n"
                f"{get_block_text(results[3])}\n"
                f"--------------------------------\n\n"
                f"[5] CNN SONUCU\n"
                f"{get_block_text(results[4])}\n"
                f"--------------------------------\n\n"
                f"[6] LSTM SONUCU\n"
                f"{get_block_text(results[5])}\n"
                f"--------------------------------\n\n"
                f"GENEL DEĞERLENDİRME\n"
                f"- En yüksek risk veren algoritma: {max_name}\n"
                f"- En düşük risk veren algoritma: {min_name}\n"
                f"- Ortalama risk skoru: {avg_str}\n"
                f"- Genel yorum: {general_comment}\n\n"
                f"KISA ÖZET\n"
                f"{summary_table}"
            )

            self.last_analysis_result = report_msg
            self.write_to_console(report_msg, color)

        except Exception as e:
            messagebox.showerror("Analiz Hatası", f"Toplu analiz gerçekleştirilemedi:\n{e}")
            self.write_to_console(f"[HATA] Toplu analiz hatası.\nDetay: {e}", COLOR_ERROR)

    def create_report(self):
        try:
            if not self.selected_image_path:
                messagebox.showwarning("Uyarı", "Lütfen önce analiz edilecek bir görüntü seçin!")
                self.write_to_console("[UYARI] Görsel seçilmediği için rapor oluşturulamadı.", COLOR_WARNING)
                return

            if not self.last_analysis_result:
                messagebox.showwarning("Uyarı", "Lütfen rapor oluşturmadan önce en az bir başarılı analiz gerçekleştirin!")
                self.write_to_console("[UYARI] Aktif analiz sonucu bulunmadığı için rapor oluşturulamadı.", COLOR_WARNING)
                return

            file_path = filedialog.asksaveasfilename(
                title="Analiz Raporunu Kaydet",
                defaultextension=".txt",
                initialfile="analiz_raporu.txt",
                filetypes=[
                    ("Metin Belgeleri", "*.txt"),
                    ("Tüm Dosyalar", "*.*")
                ]
            )

            if file_path:
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                report_content = (
                    "============================================================\n"
                    "        GÖRÜNTÜ SAHTECİLİĞİ TESPİT SİSTEMİ ANALİZ RAPORU\n"
                    "============================================================\n\n"
                    f"Rapor Tarihi / Saati   : {now_str}\n"
                    f"Analiz Edilen Görsel   : {self.selected_image_path}\n"
                    "------------------------------------------------------------\n"
                    "ANALİZ BULGULARI VE DETAYLARI:\n"
                    "------------------------------------------------------------\n"
                    f"{self.last_analysis_result}\n"
                    "------------------------------------------------------------\n"
                    "SİSTEM METADATASI VE AÇIKLAMA:\n"
                    "------------------------------------------------------------\n"
                    "Bu rapor Görüntü Sahteciliği Tespit Sistemi masaüstü uygulaması\n"
                    "tarafından otomatik üretilmiştir. İlgili analiz; görüntü üzerindeki\n"
                    "piksel oynamaları, doku bütünlüğü, gürültü analizleri, kenar geçişleri\n"
                    "veya kopyalama/taşıma benzerliklerini (klasik algoritmalar ve yapay\n"
                    "zeka modelleriyle) tarayarak sahtecilik kanıtları saptamaktadır.\n"
                    "============================================================\n"
                )

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(report_content)

                messagebox.showinfo("Başarılı", "Analiz raporu başarıyla kaydedildi!")
                self.write_to_console(f"[BAŞARILI] Rapor kaydedildi:\n{file_path}", COLOR_SUCCESS)

        except Exception as e:
            messagebox.showerror("Rapor Hatası", f"Rapor dosyası kaydedilirken bir hata oluştu:\n{e}")
            self.write_to_console(f"[HATA] Rapor kaydedilemedi.\nDetay: {e}", COLOR_ERROR)

    def reset_results(self):
        try:
            self.last_analysis_result = None
            self.write_to_console("[DURUM] Konsol temizlendi. Yeni analiz için hazır.", COLOR_MUTED)
        except Exception as e:
            messagebox.showerror("Sıfırlama Hatası", f"Konsol temizlenirken hata oluştu:\n{e}")

    def toggle_fullscreen(self, event=None):
        try:
            self.is_fullscreen = not self.is_fullscreen
            self.root.attributes("-fullscreen", self.is_fullscreen)
        except Exception as e:
            print(f"[HATA] Tam ekran geçişi başarısız: {e}")
        return "break"

    def exit_fullscreen(self, event=None):
        try:
            self.is_fullscreen = False
            self.root.attributes("-fullscreen", False)
        except Exception as e:
            print(f"[HATA] Tam ekrandan çıkış başarısız: {e}")
        return "break"

    def run(self):
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"[KRİTİK HATA] Arayüz döngüsü beklenmedik şekilde sonlandı: {e}")
