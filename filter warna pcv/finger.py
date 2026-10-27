import time
 
import cv2
import numpy as np
import mediapipe as mp
 
# --- Konfigurasi ---
DURASI_GANTI_WARNA = 5.0  # detik, sesuai permintaan: ganti warna tiap 5 detik
TEBAL_GARIS = 6
 
# Daftar warna dalam format BGR (urutan bawaan OpenCV), sesuai urutan yang diminta
DAFTAR_WARNA = [
    ("HITAM", (0, 0, 0)),
    ("PUTIH", (255, 255, 255)),
    ("MERAH", (0, 0, 255)),
    ("BIRU", (255, 0, 0)),
    ("HIJAU", (0, 255, 0)),
]
 
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
 
 
def jari_terangkat(landmark, label_tangan):
    """
    Mengembalikan list boolean [ibu_jari, telunjuk, tengah, manis, kelingking]
    berdasarkan posisi landmark tangan dari MediaPipe.
    """
    tip_ids = [4, 8, 12, 16, 20]
    status = []
 
    # Ibu jari geraknya horizontal, arahnya tergantung tangan kiri/kanan
    if label_tangan == "Right":
        status.append(landmark[tip_ids[0]].x < landmark[tip_ids[0] - 1].x)
    else:
        status.append(landmark[tip_ids[0]].x > landmark[tip_ids[0] - 1].x)
 
    # 4 jari lain: ujung jari lebih tinggi (y lebih kecil) dari sendi 2 ruas di bawahnya
    # berarti jari itu lurus/terangkat
    for i in range(1, 5):
        status.append(landmark[tip_ids[i]].y < landmark[tip_ids[i] - 2].y)
 
    return status
 
 
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[GAGAL] Tidak bisa membuka kamera. Pastikan kamera tidak dipakai aplikasi lain.")
        return
 
    kanvas = None        # gambar warna (dibuat setelah tahu ukuran frame)
    mask_gambar = None   # mask hitam-putih penanda area yang sudah digambar
    titik_sebelumnya = None
 
    idx_warna = 0
    warna_aktif = DAFTAR_WARNA[idx_warna][1]
    waktu_ganti_terakhir = time.time()
    sedang_pilih_warna = False
 
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
 
        print("\nGesture:")
        print(" - 1 jari telunjuk terangkat -> GAMBAR")
        print(" - Telapak terbuka           -> PILIH WARNA (ganti otomatis tiap 5 detik)")
        print(" - Tangan mengepal           -> HAPUS semua gambar")
        print("Tekan 'q' untuk keluar.\n")
 
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[GAGAL] Tidak bisa membaca frame dari kamera.")
                break
 
            frame = cv2.flip(frame, 1)  # efek cermin, lebih intuitif
            h, w = frame.shape[:2]
 
            if kanvas is None:
                kanvas = np.zeros((h, w, 3), dtype=np.uint8)
                mask_gambar = np.zeros((h, w), dtype=np.uint8)
 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hasil = hands.process(rgb)
 
            mode = "netral"
 
            if hasil.multi_hand_landmarks and hasil.multi_handedness:
                lm = hasil.multi_hand_landmarks[0]
                label_tangan = hasil.multi_handedness[0].classification[0].label
                status_jari = jari_terangkat(lm.landmark, label_tangan)
                jumlah_terangkat = sum(status_jari)
 
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
 
                # --- Tentukan mode dari gesture tangan ---
                if status_jari == [False, True, False, False, False]:
                    mode = "gambar"
                elif jumlah_terangkat >= 4:
                    mode = "pilih_warna"
                elif jumlah_terangkat == 0:
                    mode = "hapus"
 
                x_telunjuk = int(lm.landmark[8].x * w)
                y_telunjuk = int(lm.landmark[8].y * h)
 
                if mode == "gambar":
                    if titik_sebelumnya is not None:
                        cv2.line(kanvas, titik_sebelumnya, (x_telunjuk, y_telunjuk), warna_aktif, TEBAL_GARIS)
                        cv2.line(mask_gambar, titik_sebelumnya, (x_telunjuk, y_telunjuk), 255, TEBAL_GARIS)
                    titik_sebelumnya = (x_telunjuk, y_telunjuk)
                else:
                    titik_sebelumnya = None
 
                if mode == "hapus":
                    kanvas[:] = 0
                    mask_gambar[:] = 0
 
            else:
                titik_sebelumnya = None
 
            # --- Timer pemilihan warna: hanya berjalan selama telapak terbuka ---
            sekarang = time.time()
 
            if mode == "pilih_warna":
                if not sedang_pilih_warna:
                    # baru saja masuk mode pilih warna -> mulai hitung dari 0
                    waktu_ganti_terakhir = sekarang
                    sedang_pilih_warna = True
 
                if sekarang - waktu_ganti_terakhir >= DURASI_GANTI_WARNA:
                    idx_warna = (idx_warna + 1) % len(DAFTAR_WARNA)
                    waktu_ganti_terakhir = sekarang
 
                warna_aktif = DAFTAR_WARNA[idx_warna][1]
            else:
                sedang_pilih_warna = False
 
            # --- Gabungkan kanvas gambar dengan frame kamera pakai mask ---
            mask_inv = cv2.bitwise_not(mask_gambar)
            frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
            gambar_fg = cv2.bitwise_and(kanvas, kanvas, mask=mask_gambar)
            gabungan = cv2.add(frame_bg, gambar_fg)
 
            # --- Indikator teks mode ---
            nama_warna = DAFTAR_WARNA[idx_warna][0]
            teks_mode = {
                "gambar": "MODE: GAMBAR",
                "pilih_warna": f"MODE: PILIH WARNA -> {nama_warna}",
                "hapus": "MODE: HAPUS",
                "netral": "MODE: -",
            }[mode]
            cv2.putText(gabungan, teks_mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)
 
            # --- Kotak indikator warna aktif (pojok kanan atas) ---
            cv2.rectangle(gabungan, (w - 110, 10), (w - 10, 70), warna_aktif, -1)
            cv2.rectangle(gabungan, (w - 110, 10), (w - 10, 70), (128, 128, 128), 2)
 
            # Efek "flash": border putih tebal muncul sesaat tiap kali warna baru saja berganti
            if mode == "pilih_warna" and (sekarang - waktu_ganti_terakhir) < 0.3:
                cv2.rectangle(gabungan, (w - 115, 5), (w - 5, 75), (255, 255, 255), 4)
 
            cv2.imshow("Gambar dengan Tangan", gabungan)
 
            tombol = cv2.waitKey(1) & 0xFF
            if tombol == ord("q"):
                break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
