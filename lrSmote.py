import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Datenqualität überprüfen
# Datensatz laden
daten = pd.read_csv('creditcard.csv')

# Informationen zum Datensatz anzeigen
print("Informationen zum Datensatz:")
print(daten.info())

# Fehlende Werte prüfen
fehlende_werte = daten.isnull().sum()
print("\nFehlende Werte in jeder Spalte:")
print(fehlende_werte)

# Duplikate prüfen
duplikate = daten.duplicated().sum()
print("\nAnzahl der duplizierten Zeilen:", duplikate)

# 2. Explorative Datenanalyse (EDA)
# Grundlegende Statistiken
print("\nGrundlegende Statistiken der numerischen Merkmale:")
print(daten.describe())


# Korrelations-Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(daten.corr(), cmap='coolwarm', vmax=0.6, vmin=-0.6)
plt.title('Korrelations-Heatmap')
plt.show()

# Boxplot der Transaktionsbeträge nach Klasse
sns.boxplot(x='Class', y='Amount', data=daten)
plt.yscale('log')
plt.title('Transaktionsbeträge nach Klasse')
plt.show()

# 3. Datenvorbereitung für die logistische Regression
# Merkmale und Zielvariable trennen
X = daten.drop(['Class'], axis=1)
y = daten['Class']

# Datensatz in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Anwenden von SMOTE auf die Trainingsdaten
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Verteilung der Zielvariablen nach SMOTE anzeigen
sns.countplot(x=y_train_resampled)
plt.title('Verteilung der Zielvariablen nach SMOTE')
plt.xlabel('Klasse (0 = Nicht-Betrug, 1 = Betrug)')
plt.ylabel('Anzahl')
plt.show()

# Standardisierung der Merkmale
skalierer = StandardScaler()
X_train_skaliert = skalierer.fit_transform(X_train_resampled)
X_test_skaliert = skalierer.transform(X_test)

# 4. Logistische Regressionsmodell trainieren
log_reg = LogisticRegression()
log_reg.fit(X_train_skaliert, y_train_resampled)

# Vorhersagen auf dem Testdatensatz
y_vorhersage = log_reg.predict(X_test_skaliert)
y_vorhersage_wahrscheinlichkeit = log_reg.predict_proba(X_test_skaliert)[:, 1]

# 5. Interpretation der Ergebnisse
# Modellkoeffizienten
koeffizienten = pd.Series(log_reg.coef_[0], index=X.columns)
print("\nKoeffizienten der logistischen Regression (mit SMOTE):")
print(koeffizienten.sort_values(ascending=False))

# 6. Modellevaluierung
# Konfusionsmatrix
konfusionsmatrix = confusion_matrix(y_test, y_vorhersage)
sns.heatmap(konfusionsmatrix, annot=True, cmap='Blues', fmt='g')
plt.title('Konfusionsmatrix (mit SMOTE)')
plt.xlabel('Vorhergesagt')
plt.ylabel('Tatsächlich')
plt.show()

# Klassifikationsbericht
print("\nKlassifikationsbericht (mit SMOTE):")
print(classification_report(y_test, y_vorhersage))

# Genauigkeitsbewertung
genauigkeit = accuracy_score(y_test, y_vorhersage)
print(f"\nGenauigkeit des Modells (mit SMOTE): {genauigkeit:.2f}")

# ROC-AUC-Score
roc_auc = roc_auc_score(y_test, y_vorhersage_wahrscheinlichkeit)
print(f"ROC-AUC-Score (mit SMOTE): {roc_auc:.2f}")

# ROC-Kurve plotten
fpr, tpr, _ = roc_curve(y_test, y_vorhersage_wahrscheinlichkeit)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC-Kurve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Falsch-Positive-Rate')
plt.ylabel('True-Positive-Rate')
plt.title('Receiver Operating Characteristic (ROC) Kurve (mit SMOTE)')
plt.legend()
plt.show()
