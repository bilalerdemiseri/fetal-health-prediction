import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
import os
import sys
import xgboost
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, 'model', 'fetal_health_xgboost_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'model', 'fetal_health_scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)



def tahmin_et():
    try:
        v1 = float(entry_abnormal_long.get())
        v2 = float(entry_hist_mean.get())
        v3 = float(entry_abnormal_short.get())
        v4 = float(entry_accel.get())
        v5 = float(entry_prolonged.get())
        
        tam_veri = np.array([[133.0, v4, 0.0, 0.004, 0.0, 0.0, v5, v3, 1.3, v1, 
                              8.1, 70.0, 93.0, 164.0, 4.0, 0.0, 137.0, v2, 138.0, 18.0, 0.0]])
        
        tam_veri_scaled = scaler.transform(tam_veri)
        
        tahmin = model.predict(tam_veri_scaled)
        olasiliklar = model.predict_proba(tam_veri_scaled)[0] 
        
        yuzde_normal = olasiliklar[0] * 100
        yuzde_supheli = olasiliklar[1] * 100
        yuzde_patolojik = olasiliklar[2] * 100
        
        olasilik_metni = f"Güven Skoru: %{yuzde_normal:.1f} Normal | %{yuzde_supheli:.1f} Şüpheli | %{yuzde_patolojik:.1f} Patolojik"
        
        aciklama = "" 
        
        if tahmin[0] == 0:
            sonuc = "NORMAL"
            renk = "green"
        elif tahmin[0] == 1:
            sonuc = "ŞÜPHELİ"
            renk = "orange"
            if v1 > 50:
                aciklama = "\nAçıklama: Anormal Uzun Süreli Değişkenlik yüksek olduğu için risk tespit edildi."
        else:
            sonuc = "PATOLOJİK"
            renk = "red"
            if v5 > 0.002:
                aciklama = "\nAçıklama: Kritik 'Uzun Yavaşlamalar (Prolonged)' tespit edildiği için Kırmızı Alarm!"
            elif v1 > 70:
                aciklama = "\nAçıklama: Anormal Uzun Süreli Değişkenlik kritik seviyede!"
            
        lbl_sonuc.config(text=f"Yapay Zeka Teşhisi: {sonuc}\n\n{olasilik_metni}{aciklama}", foreground=renk)
        
    except ValueError:
        messagebox.showerror("Hata", "Lütfen tüm kutulara geçerli sayılar giriniz!")


pencere = tk.Tk()
pencere.title("FetalCare AI - Klinik Karar Destek Sistemi")
pencere.geometry("750x500")
pencere.configure(padx=20, pady=20)

tk.Label(pencere, text="Fetal Sağlık Analizi (XGBoost)", font=("Arial", 16, "bold")).pack(pady=10)


frame = ttk.Frame(pencere)
frame.pack(pady=10)

ttk.Label(frame, text="Anormal Uzun S. Değişkenlik %:").grid(row=0, column=0, sticky="w", pady=5)
entry_abnormal_long = ttk.Entry(frame)
entry_abnormal_long.insert(0, "0.0") 
entry_abnormal_long.grid(row=0, column=1, pady=5)

ttk.Label(frame, text="Histogram Ortalaması (Kalp Atışı):").grid(row=1, column=0, sticky="w", pady=5)
entry_hist_mean = ttk.Entry(frame)
entry_hist_mean.insert(0, "134.0")
entry_hist_mean.grid(row=1, column=1, pady=5)

ttk.Label(frame, text="Anormal Kısa Süreli Değişkenlik:").grid(row=2, column=0, sticky="w", pady=5)
entry_abnormal_short = ttk.Entry(frame)
entry_abnormal_short.insert(0, "45.0")
entry_abnormal_short.grid(row=2, column=1, pady=5)

ttk.Label(frame, text="İvmelenmeler (Accelerations):").grid(row=3, column=0, sticky="w", pady=5)
entry_accel = ttk.Entry(frame)
entry_accel.insert(0, "0.003")
entry_accel.grid(row=3, column=1, pady=5)

ttk.Label(frame, text="Uzun Yavaşlamalar (Prolonged):").grid(row=4, column=0, sticky="w", pady=5)
entry_prolonged = ttk.Entry(frame)
entry_prolonged.insert(0, "0.0")
entry_prolonged.grid(row=4, column=1, pady=5)


btn_tahmin = ttk.Button(pencere, text="Tıbbi Analizi Başlat", command=tahmin_et)
btn_tahmin.pack(pady=20)


lbl_sonuc = ttk.Label(pencere, text="Sonuç Bekleniyor...", font=("Arial", 14, "bold"))
lbl_sonuc.pack(pady=10)

pencere.mainloop()