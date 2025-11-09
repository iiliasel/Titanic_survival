# 🧠 Titanic Survival Prediction (XGBoost in R)

Dieses Projekt trainiert ein **XGBoost-Modell** zur Vorhersage der Überlebenswahrscheinlichkeit von Passagieren auf der Titanic.  
Der Datensatz basiert auf einem bereinigten CSV (`bereinigter_titanic_datensatz.csv`), der Merkmale wie Alter, Ticketpreis, Passagierklasse und Familiengröße enthält.

---

## 📈 Ergebnisse & Interpretation

| Kennzahl | Wert | Bedeutung |
|-----------|------|-----------|
| **Accuracy** | 69.73 % | Anteil korrekt vorhergesagter Überlebensfälle |
| **AUC (ROC)** | 77.20 % | Modell trennt Überlebende und Nicht-Überlebende deutlich besser als Zufall |
| **Baseline** | 57.47 % | Anteil der häufigeren Klasse („gestorben“) im Datensatz |

🧩 **Interpretation:**
- Das Modell reproduziert die historischen Überlebensmuster sehr gut.  
- **Höhere soziale Klasse & teurere Tickets** führten zu besseren Überlebenschancen.  
- **Jüngere Passagiere, Frauen und kleine Familien** hatten eine höhere Wahrscheinlichkeit zu überleben.  
- Das Modell ist etwas konservativ – erkennt Todesfälle sicherer als Überlebende, was typisch für den Titanic-Datensatz ist.  

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

## ⚙️ Verwendete Technologien

| Bereich | Tools & Libraries |
|----------|------------------|
| **Programmiersprache** | R (Version ≥ 4.5) |
| **Machine Learning** | `xgboost` – Gradient Boosted Decision Trees |
| **Data Preprocessing** | `caret`, `Matrix`, Basisfunktionen in R |
| **Evaluation** | `pROC` für ROC/AUC-Analyse |
| **Erklärbarkeit** | SHAP-Analyse (`predcontrib=TRUE`) zur Interpretation von Feature-Einflüssen |
| **Versionierung** | Git & GitHub |

---

## 🧠 Projektstruktur

**Dateien:**
- `ml_titanic.R` → vollständiges R-Skript (Training, Evaluation, Feature Importance & SHAP)
- `bereinigter_titanic_datensatz.csv` → bereinigter Titanic-Datensatz
- `README.md` → Projektdokumentation

---

## 🚀 Nutzung

```r
# Skript ausführen
source("ml_titanic.R")

# Vorhersage für neue Passagiere
predict_survival(df_features)
