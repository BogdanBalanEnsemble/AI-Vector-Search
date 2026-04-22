import oracledb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import array
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# ──────────────────────────────────────────────
# 1. CONFIGURARE CONEXIUNE LOCALĂ ORACLE 23ai
# ──────────────────────────────────────────────
DB_CONFIG = {
    "user": "system",
    "password": "Admin#DB1", 
    "dsn": "localhost:1522/FREE"
}

documents = [
    "How can I reset my account password?",
    "How can I reset my account password?",
    "I forgot my credentials and need to change my password.",
    "The stock market experienced a significant drop today.",
    "Major indices fell sharply as investors reacted to new economic data.",
    "I am having trouble logging in, can you help with a password reset?",
    "Wall Street closes lower amid concerns over rising inflation..."
]
labels = [f"D{i+1}" for i in range(len(documents))]

try:
    # Conectare
    conn = oracledb.connect(**DB_CONFIG)
    cursor = conn.cursor()
    print("Conectat cu succes la Oracle Database 23ai")

    # ──────────────────────────────────────────────
    # 2. PREGĂTIRE BAZĂ DE DATE (Tabel & Curățare)
    # ──────────────────────────────────────────────
    cursor.execute("BEGIN EXECUTE IMMEDIATE 'DROP TABLE doc_vectors'; EXCEPTION WHEN OTHERS THEN NULL; END;")
    
    cursor.execute("""
        CREATE TABLE doc_vectors (
            id NUMBER PRIMARY KEY,
            nume_label VARCHAR2(10),
            continut CLOB,
            v_embedding VECTOR(384, FLOAT32)
        )
        TABLESPACE USERS;
    """)
    print("Tabelul 'doc_vectors' a fost creat.")

    # ──────────────────────────────────────────────
    # 3. VECTORIZARE & INSERARE OPTIMIZATĂ
    # ──────────────────────────────────────────────
    print("⏳ Se generează vectorii de embedding...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents).tolist()

    # Folosim array pentru a trimite datele în format binar către Oracle
    for i, (doc, emb) in enumerate(zip(documents, embeddings)):
        vec_data = array.array("f", emb) 
        cursor.execute(
            "INSERT INTO doc_vectors (id, nume_label, continut, v_embedding) VALUES (:1, :2, :3, :4)",
            [i, labels[i], doc, vec_data]
        )
    conn.commit()
    print(f"Au fost inserate optimizat {len(documents)} documente.")

    # ──────────────────────────────────────────────
    # 4. CREARE INDEX VECTORIAL 
    # ──────────────────────────────────────────────
    print("⏳ Se construiește indexul vectorial...")
    # Folosim IVF (Inverted File) cu partiții de vecinătate - excelent pentru scalabilitate
    cursor.execute("""
        CREATE VECTOR INDEX doc_ivf_idx ON doc_vectors (v_embedding) 
        ORGANIZATION NEIGHBOR PARTITIONS
        DISTANCE COSINE
        WITH TARGET ACCURACY 90
        TABLESPACE USERS
    """)
    
    print("Indexul vectorial a fost creat cu succes.")

    # ──────────────────────────────────────────────
    # 5. DETECTAREA DUPLICATELOR DIRECT DIN ORACLE SQL
    # ──────────────────────────────────────────────
    DUPLICATE_THRESHOLD = 0.85
    SIMILAR_THRESHOLD   = 0.60
    
    # Calculăm distanța maximă permisă (Distanța Cosinus = 1 - Similaritate)
    MAX_DISTANCE = 1.0 - SIMILAR_THRESHOLD
    
    cursor.execute(f"""
        SELECT a.id, b.id, a.nume_label, b.nume_label, 
               1 - VECTOR_DISTANCE(a.v_embedding, b.v_embedding, COSINE) as similarity
        FROM doc_vectors a, doc_vectors b
        WHERE a.id < b.id -- Evităm compararea doc cu el însuși și duplicatele în oglindă
        AND VECTOR_DISTANCE(a.v_embedding, b.v_embedding, COSINE) <= :max_dist
        ORDER BY similarity DESC
    """, max_dist=MAX_DISTANCE)

    rezultate_sql = cursor.fetchall()
    
    duplicates = []
    similar_pairs = []
    
    print("\n🔍 REZULTATE DETECTARE (Din Oracle AI Vector Search):")
    for id_a, id_b, label_a, label_b, sim in rezultate_sql:
        if sim >= DUPLICATE_THRESHOLD:
            duplicates.append((id_a, id_b, sim))
            print(f"   [DUPLICAT] {label_a} ↔ {label_b} (Scor: {sim:.3f})")
        else:
            similar_pairs.append((id_a, id_b, sim))
            print(f"   [SIMILAR]  {label_a} ↔ {label_b} (Scor: {sim:.3f})")

    # ──────────────────────────────────────────────
    # 6. EXTRAGERE DATE PENTRU VIZUALIZARE (DASHBOARD)
    # ──────────────────────────────────────────────
    
    sim_matrix = np.zeros((len(documents), len(documents)))
    np.fill_diagonal(sim_matrix, 1.0)
    
    cursor.execute("""
        SELECT a.id, b.id, 1 - VECTOR_DISTANCE(a.v_embedding, b.v_embedding, COSINE)
        FROM doc_vectors a, doc_vectors b
        WHERE a.id != b.id
    """)
    for id_a, id_b, sim_score in cursor.fetchall():
        sim_matrix[id_a][id_b] = sim_score

    # Clustering pentru a grupa documentele pe teme
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - SIMILAR_THRESHOLD,
        metric='precomputed',
        linkage='average'
    )
    # distance matrix pentru clustering
    dist_matrix = np.clip(1 - sim_matrix, 0, None)
    np.fill_diagonal(dist_matrix, 0)
    cluster_labels = clustering.fit_predict(dist_matrix)
    unique_clusters = np.unique(cluster_labels)

    # ──────────────────────────────────────────────
    # 7. VIZUALIZARE — DASHBOARD
    # ──────────────────────────────────────────────
    plt.rcParams.update({'font.size': 13})
    fig = plt.figure(figsize=(22, 15))
    fig.suptitle("Oracle AI Vector Search — Detectarea Similarității", fontsize=20, fontweight='bold', y=0.98)

    cluster_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    doc_colors = [cluster_colors[c % len(cluster_colors)] for c in cluster_labels]

    # --- 7a. Heatmap ---
    ax1 = fig.add_subplot(2, 2, 1)
    sns.heatmap(sim_matrix, annot=True, fmt='.2f', ax=ax1, cmap="RdYlGn", xticklabels=labels, yticklabels=labels)
    ax1.set_title("Matrice Similaritate (Semantica)")

    # --- 7b. Top Perechi ---
    ax2 = fig.add_subplot(2, 2, 2)
    all_pairs = sorted([(i, j, sim_matrix[i][j]) for i in range(len(documents)) for j in range(i + 1, len(documents))], key=lambda x: -x[2])
    top_n = min(8, len(all_pairs))
    p_labels = [f"{labels[i]}↔{labels[j]}" for i, j, _ in all_pairs[:top_n]]
    p_scores = [s for _, _, s in all_pairs[:top_n]]
    
    ax2.barh(p_labels[::-1], p_scores[::-1], color='steelblue')
    ax2.axvline(DUPLICATE_THRESHOLD, color='red', linestyle='--', label='Prag Duplicat')
    ax2.set_title("Top Perechi Similare")
    ax2.legend()

    # --- 7c. Spațiu Vectorial 2D ---
    coords = PCA(n_components=2).fit_transform(np.array(embeddings))
    ax3 = fig.add_subplot(2, 2, 3)
    for idx, (x, y) in enumerate(coords):
        ax3.scatter(x, y, s=250, color=doc_colors[idx], edgecolors='black', zorder=3)
        ax3.annotate(f" {labels[idx]}", (x, y), fontweight='bold')
    ax3.set_title("Proiecție Vectorială 2D (PCA)")

    # --- 7d. Rezumat Analiză ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    text_info = f"REZUMAT ANALIZĂ ORACLE 23ai\n\n"
    text_info += f"Total documente analizate: {len(documents)}\n"
    text_info += f"Model Embedding: all-MiniLM-L6-v2 (384D)\n"
    text_info += f"Index Vectorial Utilizat: NEIGHBOR PARTITIONS (IVF)\n\n"
    text_info += f"Grupuri semantice găsite: {len(unique_clusters)}\n"
    text_info += f"Duplicate detectate (≥ {DUPLICATE_THRESHOLD}): {len(duplicates)}\n"
    text_info += f"Perechi similare (≥ {SIMILAR_THRESHOLD}): {len(similar_pairs)}\n"
    ax4.text(0.05, 0.95, text_info, transform=ax4.transAxes, fontsize=14, verticalalignment='top', family='monospace')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # SALVARE ȘI AFIȘARE
    plt.savefig("demo_oracle_result.png", dpi=150)
    print("🚀 Imaginea a fost salvată ca 'demo_oracle_result.png' în folderul curent.")
    plt.show()

except oracledb.Error as e:
    print(f"❌ Eroare Bază de date Oracle: {e}")
except Exception as e:
    print(f"❌ Eroare generală: {e}")
finally:
    if 'conn' in locals(): 
        conn.close()
        print("🔒 Conexiunea la baza de date a fost închisă.")