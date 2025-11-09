# 🧠 Titanic Survival Prediction (XGBoost in R)

Dieses Projekt trainiert ein **XGBoost-Modell** zur Vorhersage der Überlebenswahrscheinlichkeit von Passagieren auf der Titanic.  
Der Datensatz basiert auf einem bereinigten CSV (`bereinigter_titanic_datensatz.csv`), der Merkmale wie Alter, Ticketpreis, Passagierklasse und Familiengröße enthält.

---

## ⚙️ Projektaufbau

**Dateien:**
- `ml_titanic.R` → vollständiges R-Skript (Training, Evaluation, Feature Importance & SHAP)
- `bereinigter_titanic_datensatz.csv` → vorbereiteter Datensatz
- `README.md` → Projektdokumentation

---

## 📊 Modellbeschreibung

Das Modell verwendet den **XGBoost-Algorithmus (`binary:logistic`)**, um vorherzusagen, ob ein Passagier überlebt hat (`survived = 1`) oder nicht (`survived = 0`).

**Verwendete Features:**
- `pclass` – Passagierklasse (1, 2, 3)
- `age` – Alter
- `sibsp` – Geschwister/Ehepartner an Bord
- `parch` – Eltern/Kinder an Bord
- `fare` – Ticketpreis
- `sex_male` – Geschlecht (1 = männlich)
- `embarked_Q`, `embarked_S` – Einschiffungshafen (Queenstown/Southampton)

---

## 📈 Ergebnisse

| Kennzahl | Wert | Bedeutung |
|-----------|------|-----------|
| **Accuracy** | 69.73 % | Anteil korrekt vorhergesagter Überlebensfälle |
| **AUC (ROC)** | 77.20 % | Modell trennt Überlebende und Nicht-Überlebende deutlich besser als Zufall |
| **Baseline** | 57.47 % | Anteil der häufigeren Klasse („gestorben“) im Datensatz |

**Konfusionsmatrix (Test-Set):**

|               | Tatsächlich: Gestorben | Tatsächlich: Überlebt |
|----------------|------------------------|------------------------|
| **Vorhergesagt: Gestorben** | 134 | 63 |
| **Vorhergesagt: Überlebt**  | 16  | 48 |

Das Modell ist leicht **konservativ**, erkennt Todesfälle besser als Überlebende – typisch für den Titanic-Datensatz.

---

## 🔍 Wichtigste Merkmale laut Feature Importance

| Rang | Merkmal | Gain (%) | Interpretation |
|------|----------|----------|----------------|
| 1️⃣ | **fare** | 38.9 | Höherer Ticketpreis = höhere Überlebenschance |
| 2️⃣ | **age** | 29.6 | Jüngere Passagiere überlebten häufiger |
| 3️⃣ | **pclass** | 15.0 | 1. Klasse überlebte öfter als 3. Klasse |
| 4️⃣ | **sibsp** | 9.6 | Kleine Familien (0–1 Angehörige) hatten bessere Chancen |
| 5️⃣ | **parch** | 6.8 | Ähnliche Wirkung wie `sibsp`, Familiengröße wichtig |

---

## 💡 Interpretation

Das Modell reproduziert die **historischen Überlebensmuster** sehr gut:
- Frauen und Kinder überleben häufiger  
- Höhere Klassen (1. Klasse, teurere Tickets) hatten klar bessere Chancen  
- Große Familien waren beim Evakuieren im Nachteil  

---

## 🧩 Tech Stack

- **R 4.5+**
- **xgboost**, **caret**, **pROC**
- (optional) **SHAPforxgboost** oder native SHAP-Berechnung

---

## 🚀 Nutzung

```r
# Skript ausführen
source("ml_titanic.R")

# Vorhersage für neue Passagiere
predict_survival(df_features)
