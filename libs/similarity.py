from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import numpy as np


def __ngrams(string, n=3):
    """ Generate n-grams for a given string """
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def __cos_sim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M*ntop

    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))


def cos_sim(domains, threshold=0.8):
    vectorizer = TfidfVectorizer(min_df=1, analyzer=__ngrams)
    tf_idf_matrix = vectorizer.fit_transform(domains)

    matches = __cos_sim_top(tf_idf_matrix, tf_idf_matrix.transpose(), 10, threshold)
    
    non_zeros = matches.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    nr_matches = sparsecols.size

    similar_list = []

    for index in range(0, nr_matches):
        if matches.data[index] < 0.99999:
            similar_list.append({
                'domain': domains.iloc[sparserows[index]],
                'matched': domains.iloc[sparsecols[index]],
                'similairity': matches.data[index]
            })
        
    return similar_list

