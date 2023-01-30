import networkx as nx
from karateclub import DeepWalk
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score 

G = nx.karate_club_graph()
print(f"Number of nodes (club-members): {len(G.nodes)}")
nx.draw_networkx(G)

labels = []
for i in G.nodes:
    club_names = G.nodes[i]["club"]
    labels.append(1 if club_names == "Officer" else 0)

layout_pos = nx.spring_layout(G)
nx.draw_networkx(G, pos=layout_pos, node_color=labels, cmap="coolwarm")

deepwalk_model = DeepWalk(walk_number=10, walk_length=80, dimensions=124)
deepwalk_model.fit(G)

embedding = deepwalk_model.get_embedding()
print(f"Embedding Shape: {embedding.shape}")

PCA = sklearn.decomposition.PCA(n_components=2)
pca_embedding = PCA.fit_transform(embedding)
print(f"Low Dimension Embedding Shape: {pca_embedding.shape}")

plt.scatter(
    pca_embedding[:, 0], 
    pca_embedding[:, 1], 
    c=labels, s=15, 
    cmap="coolwarm"
)

x_train, x_test, y_train, y_test = train_test_split(embedding, labels, test_size=0.3)

lr_model = LogisticRegression(random_state=42).fit(x_train, y_train)

y_preds = lr_model.predict(x_test)

acc = roc_auc_score(y_test, y_preds)
print(f"AUC: {acc}")