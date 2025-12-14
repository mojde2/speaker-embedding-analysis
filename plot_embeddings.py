import argparse
import kaldiio
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


parser = argparse.ArgumentParser(description="t-SNE visualization of speaker embeddings")
parser.add_argument("--exp_dir", type=str, help=f"Path to the SCP file containing embeddings")
parser.add_argument("--speaker_ids", type=str, nargs=2, help="Two speaker IDs to visualize")
args = parser.parse_args()

exp_dir = args.exp_dir
scp_path = os.path.join(exp_dir,"embeddings/vox1/xvector.scp")
selected_ids = args.speaker_ids
output_path=os.path.join(exp_dir,"plots")

# create folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Load embeddings for selected speakers
embeddings = []
utts = []
labels = []

for utt, emb in kaldiio.load_scp_sequential(scp_path):
    speaker_id = utt.split("/")[0]  
    if speaker_id in selected_ids:
        embeddings.append(emb)
        utts.append(utt)
        labels.append(speaker_id)

embeddings = np.vstack(embeddings)
labels = np.array(labels)

# Compute centroids for each speaker
centroids = {spk: embeddings[labels == spk].mean(axis=0) for spk in selected_ids}

# Measures how tightly clustered each speaker's utterances are
intra_distances = {}
for spk in selected_ids:
    # Get all embeddings for this speaker
    spk_embeddings = embeddings[labels == spk]
    # Get the centroid for this speaker (reshape for distance function)
    spk_centroid = centroids[spk].reshape(1, -1)
    
    # Calculate the distance of each embedding to its own centroid
    distances = cosine_distances(spk_embeddings, spk_centroid)
    
    # Store the average distance (this is the intra-speaker variability metric)
    intra_distances[spk] = distances.mean()


# Measures how far apart the two speakers are from each other
centroid_vectors = np.vstack([centroids[spk] for spk in selected_ids])
cos_sim_inter = cosine_similarity(centroid_vectors)[0, 1]
# This is the inter-speaker variability metric
cos_dist_inter = cosine_distances(centroid_vectors)[0, 1] 


# Apply t-SNE
X_tsne = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

# Plot
plt.figure(figsize=(12, 10)) # Adjusted size
colors = ["blue", "green"]

centroid_tsne = []
for i, spk in enumerate(selected_ids):
    idx = labels == spk
    plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], color=colors[i], label=spk, alpha=0.6)
    
    centroid_pos = X_tsne[idx].mean(axis=0)
    centroid_tsne.append(centroid_pos)
    # Mark the centroid (the 'X')
    plt.scatter(*centroid_pos, color=colors[i], marker="x", s=200, lw=1)
    
    # Draw lines from samples to their centroid (visualizes intra-speaker variability)
    for x, y in X_tsne[idx]:
        plt.plot([x, centroid_pos[0]], [y, centroid_pos[1]], color=colors[i], alpha=0.2, lw=0.8)

# Draw the line between centroids (visualizes inter-speaker variability)
plt.plot([centroid_tsne[0][0], centroid_tsne[1][0]],
         [centroid_tsne[0][1], centroid_tsne[1][1]],
         "k--", lw=2)

# Combine all results into one string
results_text = (
    f"Inter-Speaker:\n" 
    f"  - Cosine Similarity {cos_sim_inter:.4f}\n"
    f"  - Cosine Distance {cos_dist_inter:.4f}\n\n"
    f"Intra-Speaker Distance:\n"
    f"  - {selected_ids[0]} (blue): {intra_distances[selected_ids[0]]:.4f}\n"
    f"  - {selected_ids[1]} (green): {intra_distances[selected_ids[1]]:.4f}"
)

# Place the text box in the top-right corner with a transparent background
plt.text(0.98, 0.98, results_text,
         transform=plt.gca().transAxes, # Use axis coordinates (0 to 1)
         fontsize=10,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7, edgecolor='none'))


plt.legend(loc='lower left') # Move legend to not overlap with text
plt.title(f"t-SNE Visualization: Intra- vs. Inter-Speaker Variability") # Improved title
plt.savefig(os.path.join(output_path,f"tsne_speakers_{selected_ids[0]}_{selected_ids[1]}.svg"), bbox_inches='tight')
