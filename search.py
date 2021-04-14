import numpy as np


# alpha beta are args to be adjusted
# varargs are vectors of docs of inputs from query
def rocchio_calculation(alpha, beta, query_vec, *args):
    weighted_query_np = np.multiply(alpha, query_vec)
    weighted_query_list = weighted_query_np.tolist()

    mean_np = np.mean(args, axis=0)
    weighted_mean_np = np.multiply(beta, mean_np)
    weighted_mean_list = weighted_mean_np.tolist()

    final_result = np.sum((weighted_query_list, weighted_mean_list), axis=0)
    final_result_list = final_result.tolist()
    return final_result_list

