from my_gatai import Expression_data
import numpy as np
import logging
import matplotlib.pyplot as plt


def projection_greedy(
    data: Expression_data, p_value_function, top_k=100, p_value_target=0.1
):
    p_matrix = data.p_matrix.to_numpy()

    # Center the expression profiles
    p_matrix = p_matrix - np.mean(p_matrix, axis=1)[:, None]

    tai_vector = p_matrix.sum(axis=0)
    tai_var = np.var(tai_vector)

    N = np.shape(p_matrix)[0]
    selected_gene_idxs = []

    logging.info(
        f"Step {len(selected_gene_idxs)}, Variance {tai_var}, P-value: {p_value_function(tai_var)}"
    )
    logging.info(f"tai vector: {tai_vector}")
    plt.plot(tai_vector, color="green")
    # plt.plot(tai_vector)

    while (
        len(selected_gene_idxs) < top_k and p_value_function(tai_var) < p_value_target
    ):
        ## Find the gene vector whose projection on the current tai_vector is maximal
        # Compute the dot product of each gene index against the current tai_vector
        projection_values = p_matrix.dot(tai_vector)

        # Mask the genes that have already been selected and choose the gene with the maximal projection
        mask = np.zeros(N)
        mask[selected_gene_idxs] = 1
        projection_values = np.ma.masked_array(projection_values, mask=mask)
        gene_idx = projection_values.argmax()
        selected_gene_idxs.append(gene_idx)
        gene_vector = p_matrix[gene_idx, :]
        plt.plot(gene_vector, color="red")

        ## Remove gene from TAI calculation
        gene_vector = p_matrix[gene_idx, :]
        tai_vector = tai_vector - gene_vector

        tai_var = np.var(tai_vector)

        # logging.info(
        #     f"Step {len(selected_gene_idxs)}, Variance {tai_var}, P-value: {p_value_function(tai_var)}, Removed gene: {data.full["GeneID"][gene_idx]}"
        # )

        # genes = data.full[data.full.index.isin(selected_gene_idxs)]["GeneID"]
        # data_new = data.remove_genes(genes)
        # p_matrix = data_new.p_matrix.to_numpy()
        # p_matrix = p_matrix - np.mean(p_matrix, axis=1)[:, None]
        # tai_vector = p_matrix.sum(axis=0)
        # tai_var = np.var(tai_vector)

        # logging.info(
        #     f"Step {len(selected_gene_idxs)}, Variance {tai_var}, P-value: {p_value_function(tai_var)}, Removed gene: {data.full["GeneID"][gene_idx]}"
        # )

    plt.plot(tai_vector, color="blue")

    # Get the ids of the genes from the indices
    selected_genes = data.full[data.full.index.isin(selected_gene_idxs)]
    selected_gene_ids = selected_genes["GeneID"].tolist()

    return selected_gene_ids
