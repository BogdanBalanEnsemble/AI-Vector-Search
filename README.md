# Detectarea Similarității între Documente folosind Vector Search

Acest proiect demonstrează implementarea unei soluții profesionale de identificare a documentelor duplicate și similare semantic, utilizând noile capabilități native de AI Vector Search din Oracle Database 23ai.

## Spre deosebire de metodele tradiționale, soluția mută logica de calcul a distanțelor de la nivelul aplicației direct în nucleul bazei de date, profitând de indexarea vectorială pentru performanță ridicată.

## Obiectiv

- Stocare Vectorială: Salvarea documentelor și a embedding-urilor aferente într-un format binar eficient în Oracle.
- Indexare Performantă: Utilizarea indecșilor vectoriali de tip IVF (Inverted File) pentru căutări rapide.
- Analiză în SQL: Calcularea similarității folosind funcția nativă VECTOR_DISTANCE.
- Clustering & Vizualizare: Grupare automată și generarea unui dashboard complex (PCA, Heatmap, Raport).

---

## Arhitectura Sistemului

- Fluxul de date este optimizat pentru scalabilitate:
- Text Processing: Generare embeddings (384 dimensiuni) folosind modelul all-MiniLM-L6-v2.
- Binary Upload: Conversia vectorilor în format binar (array.array) și inserare optimizată prin oracledb.
- Vector Indexing: Crearea unui index IVF în tablespace-uri cu management automat (ASSM).
- SQL Inference: Executarea interogărilor de proximitate direct pe serverul bazei de date.
- Analytics: Reducerea dimensionalității (PCA) și clustering în Python pentru interpretare.

---

## Rezultat

![Similarity Dashboard](vector-search.png)

Dashboard-ul generat conține:

- matricea de similaritate
- top perechi de documente
- reprezentare 2D a spațiului vectorial
- raport final cu rezultate

---

## Dataset

Proiectul utilizează un set mic de documente text (7 exemple), care includ:

- duplicate exacte
- parafraze (aceeași idee exprimată diferit)
- documente complet diferite (teme distincte)

Exemple:

- resetare parolă / login
- piață financiară

---

## Tehnologii utilizate

- Bază de date: Oracle Database 23ai (Free Edition).
- Limbaj: Python 3.x.
- Driver DB: oracledb
- AI/ML: sentence-transformers (HuggingFace).
- Data Science: NumPy, scikit-learn (PCA & Agglomerative Clustering).
- Vizualizare: Matplotlib & Seaborn

---

## Cum funcționează

### 1. Modelul de Date (SQL)

Tabelul este creat într-un tablespace de tip ASSM (USERS) pentru a suporta tipul de date VECTOR:

CREATE TABLE doc_vectors (
id NUMBER PRIMARY KEY,
continut CLOB,
v_embedding VECTOR(384, FLOAT32)
) TABLESPACE USERS;

---

### 2. Indexarea Vectorială

Pentru a evita căutările de tip "Brute Force", se utilizează un index de tip IVF care partiționează spațiul vectorial:

CREATE VECTOR INDEX doc_ivf_idx ON doc_vectors (v_embedding)
ORGANIZATION NEIGHBOR PARTITIONS
DISTANCE COSINE;

---

### 3. Detectarea Similarității în SQL

Se folosesc două praguri:

- **≥ 0.85** → duplicate
- **≥ 0.60** → similare semantic

Astfel:

Identificarea perechilor se face printr-o singură interogare eficientă:

SELECT 1 - VECTOR_DISTANCE(a.v_embedding, b.v_embedding, COSINE) as similarity
FROM doc_vectors a, doc_vectors b
WHERE a.id < b.id AND VECTOR_DISTANCE(...) <= :max_dist;

---

### 4. Vizualizare rezultate

Dashboard-ul conține:

- **Heatmap** – matrice de similaritate
- **Bar chart** – top perechi după scor
- **Spațiu 2D** – poziționarea documentelor în funcție de sens
- **Raport text** – rezumat final al analizei

---

## Rezultate

### Duplicate detectate

- **D1 ↔ D2 (100%)**

---

### Documente similare

- D1 ↔ D3: 77.86%
- D2 ↔ D3: 77.86%
- D1 ↔ D6: 74.64%
- D2 ↔ D6: 74.64%
- D3 ↔ D6: 62.02%
- D4 ↔ D5: 60.88%

---

### Grupuri semantice

- **Grup 1:** D4, D5 (tematică financiară)
- **Grup 2:** D1, D2, D3, D6 (login / parolă)
- **Grup 3:** D7 (document izolat)

---

## Interpretare

- Sistemul detectează corect duplicatele exacte
- Recunoaște parafraze (texte diferite, același sens)
- Grupează automat documentele pe teme
- Evidențiază relațiile semantice dintre documente

![Similarity Dashboard](compile.png)
