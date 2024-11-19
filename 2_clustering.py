from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import completeness_score, v_measure_score
import numpy as np
import pickle


# You can access the `data` folder by uncommenting the following command
# data = pickle.load(open("data/documents.p", "rb"))

def cluster_articles(data):
    vectors = np.array(data['vectors'])
    true_groups = np.array(data['group'])
    
    kmeans_100 = KMeans(n_clusters=10, random_state=2, tol=0.05, max_iter=50)
    labels_100 = kmeans_100.fit_predict(vectors)
    nobs_100 = np.bincount(labels_100)
    
    pca = PCA(n_components=10,  random_state=2)
    reduced_vectors = pca.fit_transform(vectors)
    explained_variance = pca.explained_variance_ratio_[0]
    
    kmeans_10 = KMeans(n_clusters=10, random_state=2, tol=0.05, max_iter=50)
    labels_10 = kmeans_10.fit_predict(reduced_vectors)
    nobs_10 = np.bincount(labels_10)
    
    cs_100 = completeness_score(true_groups, labels_100)
    cs_10 = completeness_score(true_groups, labels_10)
    vms_100 = v_measure_score(true_groups, labels_100)
    vms_10 = v_measure_score(true_groups, labels_10)
    
    result = {
        "nobs_100": nobs_100.tolist(),
        "nobs_10": nobs_10.tolist(),
        "pca_explained": explained_variance,
        "cs_100": cs_100,
        "cs_10": cs_10,
        "vms_100": vms_100,
        "vms_10": vms_10
    }
    
    return result


data = pickle.load(open(".\\data\\data_2\\documents.p", "rb"))

result = cluster_articles(data)

print(result)