import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score

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

# 3. Datenvorbereitung für das Random Forest Modell
# Merkmale und Zielvariable trennen
X = daten.drop(['Class'], axis=1)
y = daten['Class']

# Datensatz in Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardisierung der Merkmale
skalierer = StandardScaler()
X_train_skaliert = skalierer.fit_transform(X_train)
X_test_skaliert = skalierer.transform(X_test)

# 4. Random Forest Modell erstellen und trainieren
# Random Forest Modell initialisieren und trainieren
random_forest = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
random_forest.fit(X_train_skaliert, y_train)

# Vorhersagen auf dem Testdatensatz
y_vorhersage = random_forest.predict(X_test_skaliert)
y_vorhersage_wahrscheinlichkeit = random_forest.predict_proba(X_test_skaliert)[:, 1]

# 5. Interpretation der Ergebnisse
# Feature Importance (Bedeutung der Merkmale)
feature_importances = pd.Series(random_forest.feature_importances_, index=X.columns)
print("\nBedeutung der Merkmale (Feature Importance):")
print(feature_importances.sort_values(ascending=False))

# Visualisierung der wichtigsten Merkmale
plt.figure(figsize=(10, 6))
feature_importances.sort_values(ascending=False).head(10).plot(kind='bar', color='blue')
plt.title('Top 10 Wichtigste Merkmale laut Random Forest')
plt.xlabel('Merkmale')
plt.ylabel('Wichtigkeit')
plt.show()

# 6. Modellevaluierung
# Konfusionsmatrix
konfusionsmatrix = confusion_matrix(y_test, y_vorhersage)
sns.heatmap(konfusionsmatrix, annot=True, cmap='Blues', fmt='g')
plt.title('Konfusionsmatrix für Random Forest Modell')
plt.xlabel('Vorhergesagt')
plt.ylabel('Tatsächlich')
plt.show()

# Klassifikationsbericht
print("\nKlassifikationsbericht:")
print(classification_report(y_test, y_vorhersage))

# Genauigkeitsbewertung
genauigkeit = accuracy_score(y_test, y_vorhersage)
print(f"\nGenauigkeit des Modells: {genauigkeit:.2f}")

# ROC-AUC-Score
roc_auc = roc_auc_score(y_test, y_vorhersage_wahrscheinlichkeit)
print(f"ROC-AUC-Score: {roc_auc:.2f}")

# ROC-Kurve plotten
fpr, tpr, _ = roc_curve(y_test, y_vorhersage_wahrscheinlichkeit)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC-Kurve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('Falsch-Positive-Rate')
plt.ylabel('True-Positive-Rate')
plt.title('Receiver Operating Characteristic (ROC) Kurve für Random Forest')
plt.legend()
plt.show()