import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load artifacts ────────────────────────────────────────────────
with open(os.path.join(BASE_DIR, 'logreg_model.pkl'), 'rb') as f:
    model = pickle.load(f)

logreg_model = joblib.load(os.path.join(BASE_DIR, 'logreg_model.pkl'))
mm_scaler    = joblib.load(os.path.join(BASE_DIR, 'mm_scaler.pkl'))
# ... mungkin masih ada: nb_model, std_scaler, label_encoder, dll

# ── Konstanta (harus konsisten dengan notebook) ───────────────────
NUM_COLS = [
    'study_hours_per_day', 'attendance_percentage', 'assignment_score',
    'midterm_score', 'final_exam_score', 'participation_score', 'sleep_hours'
]
TARGET_COL   = 'Performance_Category'
DROP_FOR_MODEL = ['overall_score', 'grade', TARGET_COL]
BINARY_COLS  = ['gender', 'internet_access', 'extra_classes']
NOMINAL_COLS = ['parent_education']
CLASS_NAMES  = ['Low', 'Medium', 'High']

# ── Identitas kelompok ────────────────────────────────────
KELOMPOK = "Kelompok 3"
KELAS    = "SI-48-01"
PRODI    = "S1 Sistem Informasi"
KAMPUS   = "Universitas Telkom"
ANGGOTA  = [
    {"nama": "Malky Sudrajat Asshidiq", "nim": "102022400204"},
    {"nama": "Sitti Naurah bauw", "nim": "102022430071"},
    {"nama": "Mia Stacia Adelia Effendi", "nim": "102022400134"},
]

# ── Label fitur agar mudah dibaca pada narasi ─────────────
FEATURE_LABELS = {
    'study_hours_per_day':   'jam belajar per hari',
    'attendance_percentage': 'persentase kehadiran',
    'assignment_score':      'nilai tugas',
    'midterm_score':         'nilai UTS',
    'final_exam_score':      'nilai UAS',
    'participation_score':   'nilai partisipasi',
    'sleep_hours':           'jam tidur per hari',
    'gender_Male':           'jenis kelamin laki-laki',
    'gender_Female':         'jenis kelamin perempuan',
    'internet_access_Yes':   'memiliki akses internet',
    'internet_access_No':    'tidak memiliki akses internet',
    'extra_classes_Yes':     'mengikuti kelas tambahan',
    'extra_classes_No':      'tidak mengikuti kelas tambahan',
    'parent_education_High School': 'pendidikan orang tua (SMA)',
    'parent_education_Bachelor':    'pendidikan orang tua (S1)',
    'parent_education_Master':      'pendidikan orang tua (S2)',
    'parent_education_PhD':         'pendidikan orang tua (S3)',
}

# ── Load & prep data untuk metrik evaluasi ────────────────────────
@st.cache_data
def load_eval_data():
    df_raw = pd.read_csv('student_performance_data.csv')
    df = df_raw.drop(columns=['student_id'])

    # Outlier handling (IQR) — sama persis dengan notebook
    df_clean = df.copy()
    for col in NUM_COLS:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[
            (df_clean[col] >= Q1 - 1.5 * IQR) &
            (df_clean[col] <= Q3 + 1.5 * IQR)
        ]
    df = df_clean.reset_index(drop=True)

    # Binning — quantile sama dengan notebook
    q33 = df['overall_score'].quantile(0.33)
    q66 = df['overall_score'].quantile(0.66)
    df[TARGET_COL] = pd.cut(
        df['overall_score'],
        bins=[-np.inf, q33, q66, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    # Encoding
    df_encoded = pd.get_dummies(df, columns=BINARY_COLS + NOMINAL_COLS, drop_first=False)
    df_encoded[NUM_COLS] = mm_scaler.transform(df_encoded[NUM_COLS])

    drop_cols = [col for col in DROP_FOR_MODEL + ['grade'] if col in df_encoded.columns]
    X = df_encoded.drop(columns=drop_cols)
    X = X.astype({col: int for col in X.select_dtypes(include='bool').columns})
    X = X.reindex(columns=model_cols, fill_value=0)
    y = df_encoded[TARGET_COL]

    return X, y


# ── Fungsi pembuat narasi penjelasan prediksi ─────────────
def buat_narasi(pred_label, proba, raw_vals, coef_df):
    """Menghasilkan paragraf naratif yang menjelaskan alasan di balik prediksi."""

    def level_skor(v):
        if v >= 70: return "tergolong tinggi"
        if v >= 55: return "berada pada kisaran sedang"
        return "tergolong rendah"

    def level_jam_belajar(v):
        if v >= 7: return "cukup intens"
        if v >= 4: return "cukup memadai"
        return "tergolong rendah"

    def level_hadir(v):
        if v >= 80: return "tergolong tinggi"
        if v >= 60: return "berada pada kisaran sedang"
        return "tergolong rendah"

    tugas   = raw_vals['assignment_score']
    uts     = raw_vals['midterm_score']
    uas     = raw_vals['final_exam_score']
    partisi = raw_vals['participation_score']
    jam     = raw_vals['study_hours_per_day']
    hadir   = raw_vals['attendance_percentage']

    rata_akademik = np.mean([tugas, uts, uas, partisi])
    if rata_akademik >= 70:
        ringkas_akademik = "secara umum tergolong tinggi"
    elif rata_akademik >= 55:
        ringkas_akademik = "secara umum berada pada kisaran menengah"
    else:
        ringkas_akademik = "secara umum tergolong rendah"

    # Tiga faktor model paling berpengaruh, diterjemahkan ke bahasa yang mudah dibaca
    faktor_utama = []
    for fitur in coef_df['Fitur'].head(3):
        faktor_utama.append(FEATURE_LABELS.get(fitur, fitur))
    faktor_str = ", ".join(faktor_utama)

    # Paragraf 1 — keputusan, keyakinan, dan profil akademik
    p1 = (
        f"Model memprediksi siswa ini termasuk dalam kategori performa **{pred_label}** "
        f"dengan tingkat keyakinan sebesar {proba:.1%}. Prediksi tersebut terutama "
        f"didasarkan pada profil nilai akademik siswa, yaitu nilai tugas sebesar {tugas:.0f} "
        f"({level_skor(tugas)}), nilai UTS sebesar {uts:.0f} ({level_skor(uts)}), "
        f"nilai UAS sebesar {uas:.0f} ({level_skor(uas)}), serta nilai partisipasi sebesar "
        f"{partisi:.0f} ({level_skor(partisi)}). Jika dilihat secara keseluruhan, capaian "
        f"akademik siswa ini {ringkas_akademik}."
    )

    # Paragraf 2 — faktor pendukung + faktor dominan model
    p2 = (
        f"Selain capaian akademik, faktor pendukung lain turut diperhitungkan, antara lain "
        f"jam belajar sebesar {jam:.1f} jam per hari ({level_jam_belajar(jam)}) dan tingkat "
        f"kehadiran sebesar {hadir:.0f}% ({level_hadir(hadir)}). Berdasarkan analisis kontribusi "
        f"model, faktor yang paling mendorong siswa diklasifikasikan ke kategori {pred_label} "
        f"adalah {faktor_str}. Hal ini sejalan dengan logika data, karena kategori performa pada "
        f"dasarnya terbentuk dari akumulasi nilai-nilai akademik tersebut."
    )

    # Paragraf 3 — penutup sesuai kategori
    if pred_label == "High":
        p3 = ("Kombinasi capaian yang baik pada sebagian besar aspek inilah yang membuat siswa "
              "diprediksi memiliki performa tinggi.")
    elif pred_label == "Medium":
        p3 = ("Capaian yang berada pada kisaran menengah dan belum konsisten menonjol di seluruh "
              "aspek membuat siswa diprediksi memiliki performa sedang.")
    else:
        p3 = ("Capaian yang relatif rendah pada beberapa aspek membuat siswa diprediksi berisiko "
              "memiliki performa rendah, sehingga dapat menjadi perhatian pihak akademik untuk "
              "pendampingan atau intervensi lebih lanjut.")

    return p1, p2, p3


X_eval, y_eval = load_eval_data()

y_pred_eval = model.predict(X_eval)
y_prob_eval = model.predict_proba(X_eval)

acc  = accuracy_score(y_eval, y_pred_eval)
prec = precision_score(y_eval, y_pred_eval, average='weighted')
rec  = recall_score(y_eval, y_pred_eval, average='weighted')
y_bin = label_binarize(y_eval, classes=[0, 1, 2])

# ── Sidebar identitas kelompok ────────────────────────────
with st.sidebar:
    st.header("Identitas Kelompok")
    st.markdown(f"**{KELOMPOK}**")
    st.markdown(f"Kelas: **{KELAS}**")
    st.markdown(f"{PRODI} — {KAMPUS}")
    st.divider()
    st.subheader("Anggota Kelompok")
    for i, a in enumerate(ANGGOTA, 1):
        st.markdown(f"{i}. **{a['nama']}** — {a['nim']}")

# ── Layout ────────────────────────────────────────────────────────
st.title("Student Performance Prediction")
st.markdown("Prediksi kategori performa siswa menggunakan Logistic Regression.")
st.divider()

# ── Metrik evaluasi ───────────────────────────────────────────────
st.subheader("Evaluasi Model")
col1, col2, col3 = st.columns(3)
col1.metric("Accuracy",  f"{acc:.2f}")
col2.metric("Precision", f"{prec:.2f}")
col3.metric("Recall",    f"{rec:.2f}")

# ── Visualisasi ───────────────────────────────────────────────────
st.subheader("Visualisasi")
chart_option = st.selectbox("Pilih grafik:", ["ROC AUC Curve", "Confusion Matrix"])

if chart_option == "ROC AUC Curve":
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob_eval[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC AUC Curve — Logistic Regression')
    ax.legend()
    st.pyplot(fig)
else:
    cm = confusion_matrix(y_eval, y_pred_eval)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix — Logistic Regression')
    st.pyplot(fig)

st.divider()

# ── Form prediksi ─────────────────────────────────────────────────
st.subheader("Prediksi Performa Siswa")
st.markdown("Masukkan data siswa sesuai nilai aslinya (sebelum normalisasi).")

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        gender          = st.selectbox("Gender", ["Male", "Female"])
        study_hours     = st.number_input("Jam Belajar per Hari", min_value=0.0, max_value=24.0, step=0.5, value=6.0)
        attendance      = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0, value=80.0)
        assignment      = st.number_input("Assignment Score (0-100)", min_value=0.0, max_value=100.0, step=1.0, value=70.0)
        midterm         = st.number_input("Midterm Score (0-100)", min_value=0.0, max_value=100.0, step=1.0, value=70.0)

    with col2:
        final_exam      = st.number_input("Final Exam Score (0-100)", min_value=0.0, max_value=100.0, step=1.0, value=70.0)
        participation   = st.number_input("Participation Score (0-100)", min_value=0.0, max_value=100.0, step=1.0, value=70.0)
        sleep_hours     = st.number_input("Jam Tidur per Hari", min_value=0.0, max_value=24.0, step=0.5, value=7.0)
        internet_access = st.selectbox("Internet Access", ["Yes", "No"])
        extra_classes   = st.selectbox("Extra Classes", ["Yes", "No"])
        parent_edu      = st.selectbox("Parent Education", ["High School", "Bachelor", "Master", "PhD"])

    submit = st.form_submit_button("Prediksi")

if submit:
    # ── Simpan nilai asli (sebelum normalisasi) untuk narasi
    raw_vals = {
        'study_hours_per_day':   study_hours,
        'attendance_percentage': attendance,
        'assignment_score':      assignment,
        'midterm_score':         midterm,
        'final_exam_score':      final_exam,
        'participation_score':   participation,
        'sleep_hours':           sleep_hours,
    }

    # Buat dataframe input dalam skala asli
    input_raw = pd.DataFrame([{
        'study_hours_per_day':   study_hours,
        'attendance_percentage': attendance,
        'assignment_score':      assignment,
        'midterm_score':         midterm,
        'final_exam_score':      final_exam,
        'participation_score':   participation,
        'sleep_hours':           sleep_hours,
        'gender':                gender,
        'internet_access':       internet_access,
        'extra_classes':         extra_classes,
        'parent_education':      parent_edu
    }])

    # Normalisasi NUM_COLS pakai scaler yang sama
    input_raw[NUM_COLS] = mm_scaler.transform(input_raw[NUM_COLS])

    # One-hot encoding
    input_encoded = pd.get_dummies(input_raw, columns=BINARY_COLS + NOMINAL_COLS, drop_first=False)

    # Align kolom dengan model
    input_aligned = input_encoded.reindex(columns=model_cols, fill_value=0)
    input_aligned = input_aligned.astype({
        col: int for col in input_aligned.select_dtypes(include='bool').columns
    })

    # Prediksi
    pred       = model.predict(input_aligned)[0]
    pred_proba = model.predict_proba(input_aligned)[0]
    pred_label = CLASS_NAMES[pred]

    # Tampilkan hasil
    st.success(f"Prediksi Kategori Performa: **{pred_label}**")

    st.subheader("Probabilitas per Kategori")
    proba_df = pd.DataFrame({
        'Kategori':    CLASS_NAMES,
        'Probabilitas': pred_proba
    }).sort_values('Probabilitas', ascending=False)

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#2196F3' if c == pred_label else '#B0BEC5' for c in proba_df['Kategori']]
    ax.barh(proba_df['Kategori'], proba_df['Probabilitas'], color=colors)
    ax.set_xlabel('Probabilitas')
    ax.set_xlim(0, 1)
    ax.set_title('Distribusi Probabilitas Prediksi')
    for i, (val, cat) in enumerate(zip(proba_df['Probabilitas'], proba_df['Kategori'])):
        ax.text(val + 0.01, i, f'{val:.1%}', va='center')
    st.pyplot(fig)

    # Hitung kontribusi fitur
    coef_df = pd.DataFrame({
        'Fitur':       model_cols,
        'Koefisien':   model.coef_[pred]
    }).reindex(columns=['Fitur', 'Koefisien'])

    input_vals = input_aligned.iloc[0]
    coef_df['Kontribusi'] = coef_df['Koefisien'] * input_vals.values
    coef_df = coef_df.reindex(columns=['Fitur', 'Koefisien', 'Kontribusi'])
    coef_df = coef_df.sort_values('Kontribusi', ascending=False).head(5)

    # ── Narasi penjelasan prediksi ───────────────────────
    st.subheader("Penjelasan Hasil Prediksi")
    p1, p2, p3 = buat_narasi(pred_label, pred_proba[pred], raw_vals, coef_df)
    st.markdown(p1)
    st.markdown(p2)
    st.markdown(p3)

    # Tabel faktor (tetap dipertahankan sebagai pelengkap)
    st.subheader("Faktor yang Mempengaruhi Prediksi")
    st.markdown(f"**5 fitur dengan kontribusi terbesar ke kelas '{pred_label}':**")
    st.dataframe(coef_df.reset_index(drop=True), use_container_width=True)
