import math
import numpy as np


'''
# input
#    - pred_u: 예측 값으로 정렬 된 item index
#    - target_u: test set의 item index
#    - top_k: top-k에서의 k 값
'''
def compute_metrics(pred_u, target_u, top_k):
    pred_k = pred_u[:top_k] # 예측된 상위 k개
    num_target_items = len(target_u)

    hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_u]
    num_hits = len(hits_k)

    idcg_k = 0.0
    for i in range(1, min(num_target_items, top_k) + 1):
        idcg_k += 1 / math.log(i + 1, 2)

    dcg_k = 0.0
    for idx, item in hits_k:
        dcg_k += 1 / math.log(idx + 1, 2)

    prec_k = num_hits / top_k
    recall_k = num_hits / min(num_target_items, top_k)
    ndcg_k = dcg_k / idcg_k

    '''
    Implement RR@K and AP@K in here
    '''
    # ========================= EDIT HERE ========================
    if len(hits_k) == 0: # 맟준게 없을 경우
        rr_k = 0
    else:
        rr_k =  1 / hits_k[0][0]
    
    if len(hits_k) == 0: # 맞춘게 없을 경우
        ap_k = 0
    else:
        precs = 0
        for i, hit in enumerate(hits_k):
            prec = (i + 1) / hit[0] 
            precs += prec
        ap_k = precs / num_hits
    # ============================================================
    return prec_k, recall_k, ndcg_k, rr_k, ap_k

'''
You can implement metrics using follow functions if you want
'''
def get_rr_k(pred_u, target_u, top_k):
    # ========================= EDIT HERE ========================
    rr_k = -1
    # ============================================================
    return rr_k

def get_ap_k(pred_u, target_u, top_k):
    # ========================= EDIT HERE ========================
    ap_k = -1
    # ============================================================
    return ap_k