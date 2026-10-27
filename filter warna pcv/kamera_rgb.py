import cv2
import numpy as np

# Mengakses webcam laptop (0 biasanya adalah kamera bawaan)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Kamera tidak dapat diakses.")
  exit()

# Mode filter awal
filter_mode = 'normal'

print("--- KONTROL FILTER ---")
print("Tekan 'n' : Normal")
print("Tekan 'r' : Filter Merah (Red)")
print("Tekan 'g' : Filter Hijau (Green)")
print("Tekan 'b' : Filter Biru (Blue)")
print("Tekan 'q' : Keluar dari aplikasi")

while True:
  ret, frame = cap.read()
  if not ret:
    print("Gagal mengambil gambar dari kamera.")
    break

  # Membalikkan gambar secara horizontal (efek cermin agar gerakan tidak terbalik)
  frame = cv2.flip(frame, 1)

  # Membuat salinan frame agar tidak mengubah data asli secara langsung
  output = frame.copy()

  # Format warna OpenCV adalah BGR (Blue, Green, Red)
  if filter_mode == 'red':
    # Matikan channel Blue (0) dan Green (1), sisakan Red (2)
    output[:, :, 0] = 0  # Blue = 0
    output[:, :, 1] = 0  # Green = 0

  elif filter_mode == 'green':
    # Matikan channel Blue (0) dan Red (2), sisakan Green (1)
    output[:, :, 0] = 0  # Blue = 0
    output[:, :, 2] = 0  # Red = 0

  elif filter_mode == 'blue':
    # Matikan channel Green (1) dan Red (2), sisakan Blue (0)
    output[:, :, 1] = 0  # Green = 0
    output[:, :, 2] = 0  # Red = 0

  # Menambahkan label teks filter aktif di pojok atas layar
  cv2.putText(
      output,
      f"Filter: {filter_mode.upper()}",
      (20, 40),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 255, 255),
      2,
      cv2.LINE_AA,
  )

  # Menampilkan hasil ke jendela video
  cv2.imshow("Filter Kamera RGB", output)

  # Membaca tombol keyboard yang ditekan pengguna
  key = cv2.waitKey(1) & 0xFF
  if key == ord('q'):
    break  # Keluar dari loop
  elif key == ord('n'):
    filter_mode = 'normal'
  elif key == ord('r'):
    filter_mode = 'red'
  elif key == ord('g'):
    filter_mode = 'green'
  elif key == ord('b'):
    filter_mode = 'blue'

# Membersihkan resources
cap.release()
cv2.destroyAllWindows()