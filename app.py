
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from pathlib import Path

app = Flask(__name__)

# Baca dataset dari folder project
LOKASI_CSV = Path(__file__).parent / "german_credit_data.csv"

data = pd.read_csv(LOKASI_CSV)

# Periksa kolom dataset
kolom_wajib = ["Credit amount", "Risk"]

for kolom in kolom_wajib:
    if kolom not in data.columns:
        raise ValueError(
            f"Kolom '{kolom}' tidak ditemukan di CSV. "
            f"Kolom yang tersedia: {list(data.columns)}"
        )

# Bersihkan data yang dibutuhkan untuk simulasi
data = data[kolom_wajib].copy()
data["Credit amount"] = pd.to_numeric(
    data["Credit amount"], errors="coerce"
)
data = data.dropna(subset=["Credit amount", "Risk"])

# Risk Bad diasumsikan gagal bayar, sehingga jumlah kredit
# menjadi nilai kerugian. Risk Good diasumsikan tidak rugi.
data["Risk"] = data["Risk"].astype(str).str.strip().str.lower()

data = data[data["Risk"].isin(["good", "bad"])]

data["Kerugian"] = np.where(
    data["Risk"] == "bad",
    data["Credit amount"],
    0
)

nilai_kerugian = data["Kerugian"].to_numpy()


@app.route("/", methods=["GET", "POST"])
def beranda():
    # Nilai awal pada form
    jumlah_pinjaman = 100
    jumlah_simulasi = 5000
    ambang = 100000

    # Metrik belum ada sebelum tombol simulasi ditekan
    hasil = None
    error = None

    if request.method == "POST":
        try:
            jumlah_pinjaman = int(
                request.form.get("jumlah_pinjaman", 100)
            )
            jumlah_simulasi = int(
                request.form.get("jumlah_simulasi", 5000)
            )
            ambang = float(request.form.get("ambang", 100000))

            if jumlah_pinjaman < 1:
                raise ValueError("Jumlah pinjaman minimal 1.")

            if jumlah_simulasi < 1:
                raise ValueError("Jumlah simulasi minimal 1.")

            if ambang < 0:
                raise ValueError("Ambang kerugian tidak boleh negatif.")

            if len(nilai_kerugian) == 0:
                raise ValueError("Dataset tidak memiliki data Risk good/bad.")

            # Jalankan bootstrap Monte Carlo:
            # tiap skenario mengambil N pinjaman dari data,
            # dengan pengembalian (sampling with replacement).
            rng = np.random.default_rng()

            total_kerugian = np.empty(jumlah_simulasi)

            for i in range(jumlah_simulasi):
                sampel = rng.choice(
                    nilai_kerugian,
                    size=jumlah_pinjaman,
                    replace=True
                )
                total_kerugian[i] = sampel.sum()

            # Ringkasan hasil simulasi
            hasil = {
                "rata_rata": float(total_kerugian.mean()),
                "minimum": float(total_kerugian.min()),
                "maksimum": float(total_kerugian.max()),
                "probabilitas": float(
                    np.mean(total_kerugian > ambang)
                ),
                "jumlah_skenario": jumlah_simulasi
            }

        except ValueError as e:
            error = str(e)

    return render_template(
        "index.html",
        jumlah_pinjaman=jumlah_pinjaman,
        jumlah_simulasi=jumlah_simulasi,
        ambang=ambang,
        hasil=hasil,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
