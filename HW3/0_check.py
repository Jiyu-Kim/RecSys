from metrics import compute_metrics
import numpy as np

pred_u = np.array([1, 2, 3, 4, 5], dtype=np.float32)
target_u = np.array([3, 5], dtype=np.float32)
top_k = 5

prec_k, recall_k, ndcg_k, rr_k, ap_k = compute_metrics(pred_u, target_u, top_k)
print(f'prec_k {prec_k}, recall_k {recall_k}, ndcg_k {ndcg_k}, rr_k {rr_k}, ap_k {ap_k}')

try:
    assert (0.3333333333333333 - rr_k) < 0.00001
    print("rr_k is correct")
except:
    print("rr_k is wrong")

try:
    assert (0.3666666666666667 - ap_k) < 0.00001
    print("ap_k is correct")
except:
    print("ap_k is wrong")
