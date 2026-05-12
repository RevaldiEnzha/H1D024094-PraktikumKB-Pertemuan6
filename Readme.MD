# Jaringan Syaraf Tiruan (JST) — Praktikum 6

Repositori ini berisi implementasi dua algoritma Jaringan Syaraf Tiruan (JST) dalam Python, yaitu **Perceptron** dan **Backpropagation**.

---

## Daftar File
| File | Keterangan |
|------|------------|
| `Perceptron.py` | Definisi kelas Perceptron |
| `Perceptron_or.py` | Eksekusi Perceptron untuk masalah OR |
| `Backpropagation.py` | Definisi kelas Backpropagation |
| `Backpropagation_xor.py` | Eksekusi Backpropagation untuk masalah XOR |

---

## Cara Menjalankan
Pastikan Python dan library sudah terpasang:

```bash
pip install numpy matplotlib
```

Untuk menjalankan masing-masing program:

```bash
# Menjalankan Perceptron (masalah OR)
python Perceptron_or.py

# Menjalankan Backpropagation (masalah XOR)
python Backpropagation_xor.py
```

---

## 1. Perceptron

Perceptron adalah model JST paling sederhana yang hanya memiliki satu layer. Model ini cocok untuk permasalahan yang datanya bisa dipisahkan secara linier, seperti masalah logika OR.

### `Perceptron.py`

File ini mendefinisikan kelas `Perceptron` beserta semua fungsi yang dibutuhkan untuk proses pelatihan.

---

#### `class Perceptron`

```python
class Perceptron:
```

Kelas utama yang menampung semua logika algoritma Perceptron, mulai dari inisialisasi parameter hingga pelatihan model.

---

#### `__init__(self, alpha=0.1, epoch=10)`

```python
def __init__(self, alpha=0.1, epoch=10):
    self.alpha = alpha
    self.epoch = epoch
```

Konstruktor kelas. Dijalankan pertama kali saat objek `Perceptron` dibuat. Bertugas menyimpan dua parameter:
- `alpha` — learning rate, yaitu seberapa besar langkah pembaruan bobot setiap iterasi.
- `epoch` — jumlah maksimum putaran pelatihan yang diizinkan.

---

#### `weighted_sum(self, X)`

```python
def weighted_sum(self, X):
    return np.dot(X, self.w_[1:]) + self.w_[0]
```

Menghitung nilai `y_in` (net input), yaitu hasil perkalian antara setiap input dengan bobotnya masing-masing, kemudian ditambah bias. Ini adalah operasi sebelum fungsi aktivasi diterapkan.

---

#### `predict(self, X)`

```python
def predict(self, X):
    return np.where(self.weighted_sum(X) >= 0.0, 1, -1)
```

Menerapkan fungsi aktivasi bipolar. Jika nilai `y_in` lebih besar atau sama dengan 0, output adalah `1`; jika tidak, output adalah `-1`. Fungsi ini digunakan untuk menghasilkan prediksi dari model.

---

#### `plot_decision_boundary(self, X, t, epoch)`

```python
def plot_decision_boundary(self, X, t, epoch):
```

Membuat visualisasi garis pemisah (decision boundary) pada setiap epoch. Titik-titik data input ditampilkan dengan warna berbeda sesuai kelasnya, dan garis pemisah digambar berdasarkan nilai bobot dan bias yang sudah diperbarui. Untuk melihat bagaimana model belajar dari waktu ke waktu.

---

#### `fit(self, X, t)`

```python
def fit(self, X, t):
```

Fungsi utama pelatihan Perceptron. Proses yang terjadi di dalamnya:

1. Inisialisasi bobot dan bias ke nilai 0.
2. Iterasi epoch. Proses diulang sebanyak `epoch` kali atau sampai kondisi berhenti terpenuhi.
3. Untuk setiap pasang input dan target:
   - Hitung prediksi dengan `predict()`.
   - Hitung error (`target - prediksi`).
   - Perbarui bobot dan bias menggunakan Delta Rule
4. Tampilkan garis pemisah per epoch.
5. Kondisi berhenti: jika Sum Square Error (SSE) = 0 atau epoch maksimum tercapai.
6. Semua hasil perhitungan disimpan ke file `HasilPerceptron.txt`.

---

### `Perceptron_or.py`

File ini adalah file eksekusi untuk menjalankan model Perceptron pada masalah logika OR dengan data bipolar.

```python
X = np.array([[1,1], [1,-1], [-1,1], [-1,-1]])
t = np.array([[1], [1], [1], [-1]])
```

Matriks `X` adalah data input (4 kombinasi), dan `t` adalah target output sesuai tabel kebenaran OR bipolar. Model diinisialisasi dengan `alpha=0.1` dan `epoch=10`, lalu langsung dilatih dengan memanggil `model.fit(X, t)`.

---

## 2. Backpropagation

Backpropagation adalah algoritma JST yang mampu menyelesaikan permasalahan yang tidak bisa dipisahkan secara linier, seperti masalah logika XOR.

### `Backpropagation.py`

File ini mendefinisikan kelas `Backpropagation` dengan seluruh mekanisme forward dan backward propagation.

---

#### `class Backpropagation`

```python
class Backpropagation:
```

Kelas utama yang menampung semua komponen algoritma Backpropagation, termasuk inisialisasi bobot acak, fungsi aktivasi, dan proses pelatihan dua arah (maju dan mundur).

---

#### `__init__(self, alpha, epoch, target_error)`

```python
def __init__(self, alpha, epoch, target_error):
    self.alpha = alpha
    self.epoch = epoch
    self.target_error = target_error

    self.n_input = 2
    self.n_hidden = 2
    self.n_output = 1

    self.w_hidden = np.random.rand(self.n_input, self.n_hidden)
    self.b_hidden = np.random.rand(1, self.n_hidden)
    self.w_output = np.random.rand(self.n_hidden, self.n_output)
    self.b_output = np.random.rand(1, self.n_output)
```

Konstruktor kelas. Selain menyimpan parameter pelatihan (`alpha`, `epoch`, `target_error`), konstruktor ini juga mendefinisikan arsitektur jaringan:
- `n_input = 2` — jumlah neuron input.
- `n_hidden = 2` — jumlah neuron pada hidden layer.
- `n_output = 1` — jumlah neuron output.

Bobot dan bias untuk hidden layer maupun output layer diinisialisasi secara **acak** menggunakan `np.random.rand`, agar proses pembelajaran bisa berjalan dari titik awal yang berbeda-beda.

---

#### `bi_sigmoid(self, x)`

```python
def bi_sigmoid(self, x):
    return np.tanh(x)
```

Fungsi aktivasi sigmoid bipolar, atau biasa dikenal **tanh**. Menghasilkan nilai output di antara -1 dan 1.

---

#### `deriv_bi_sigmoid(self, x)`

```python
def deriv_bi_sigmoid(self, x):
    return 1 - x**2
```

Turunan dari fungsi tanh. Digunakan saat proses backward propagation untuk menghitung seberapa besar penyesuaian bobot yang perlu dilakukan. Asumsinya, `x` yang dimasukkan adalah **hasil output dari fungsi tanh**, bukan input aslinya.

---

#### `plot_error(self, x, epoch)`

```python
def plot_error(self, x, epoch):
```

Membuat grafik penurunan error (SSE) dari epoch pertama hingga epoch terakhir saat pelatihan berhenti. Grafik ini memvisualisasikan seberapa cepat dan efektif model belajar. Terdapat anotasi yang menunjukkan nilai error akhir pada epoch terakhir.

---

#### `fit(self, X, t)`

```python
def fit(self, X, t):
```

Fungsi utama pelatihan Backpropagation. Proses yang terjadi di dalamnya:

**Forward Propagation** (arah maju — dari input ke output):
1. Hitung `h_in` — perkalian input dengan bobot hidden layer, ditambah bias hidden.
2. Aktivasi `h_in` dengan fungsi tanh untuk mendapatkan output hidden layer (`h`).
3. Hitung `y_in` — perkalian `h` dengan bobot output layer, ditambah bias output.
4. Aktivasi `y_in` dengan fungsi tanh untuk mendapatkan output akhir (`y`).

**Backward Propagation** (arah mundur — dari output ke input):
1. Hitung `error = target - y`.
2. Hitung delta output layer: `d_y = error × tanh'(y)`.
3. Hitung error yang diteruskan ke hidden layer: `error_h = d_y × w_output^T`.
4. Hitung delta hidden layer: `d_h = error_h × tanh'(h)`.

**Pembaruan Bobot dan Bias:**
- Bobot dan bias output layer diperbarui berdasarkan `d_y`.
- Bobot dan bias hidden layer diperbarui berdasarkan `d_h`.

**Kondisi berhenti**: jika rata-rata SSE sudah di bawah `target_error`, atau epoch maksimum sudah tercapai. Semua hasil perhitungan disimpan ke `hasilBackpropagation.txt`, dan grafik penurunan error ditampilkan di akhir.

---

### `Backpropagation_xor.py`

File ini adalah file eksekusi untuk menjalankan model Backpropagation pada masalah logika XOR dengan data bipolar.

```python
X = np.array([
    [1, 1],
    [1, -1],
    [-1, 1],
    [-1, -1]
])

t = np.array([
    [-1],
    [1],
    [1],
    [-1]
])
```

Matriks `X` adalah 4 kombinasi input XOR, dan `t` adalah target output sesuai tabel kebenaran XOR bipolar. Model diinisialisasi dengan `alpha=0.3`, `epoch=1000`, dan `target_error=0.001`, lalu dilatih dengan memanggil `model.fit(X, t)`.

---

## Output Program

- **`HasilPerceptron.txt`** — log perhitungan detail setiap iterasi Perceptron, termasuk bobot akhir.
- **`hasilBackpropagation.txt`** — log perhitungan detail setiap iterasi Backpropagation, termasuk bobot akhir.
- **Grafik matplotlib** — ditampilkan langsung saat program berjalan (decision boundary untuk Perceptron, grafik SSE untuk Backpropagation).