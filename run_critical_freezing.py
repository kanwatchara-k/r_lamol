#!/usr/bin/env python3

import os, sys, json, logging, csv
import argparse
import itertools, math
from itertools import chain
from rationale_benchmark.utils import load_documents, load_datasets, annotations_from_jsonl, Annotation
import numpy as np
from scipy import stats
from pathlib import Path 
import torch
from pytorch_transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel #, GPT2Model, 
from typing import Any, Callable, Dict, List, Set, Tuple
from sklearn.metrics import auc, precision_recall_curve, jaccard_score, f1_score
from tqdm import tqdm
from collections import Counter
from pathos.pools import ProcessPool
from rationale_benchmark.utils import (
    annotations_from_jsonl,
    load_flattened_documents
)
from scipy.spatial.distance import pdist
import functools , time
from datetime import datetime
# from pympler import tracker
pool = ProcessPool(nodes=12)

model_name = 'gpt2'

def _avg_auprc(truths, preds):
    if len(preds) == 0:
        return 0.0
    assert len(truth.keys() and preds.keys()) == len(truth.keys())
    aucs = []
    for k, true in truth.items():
        pred = preds[k]
        aucs.append(_auprc(true, pred))
    return np.average(aucs)

def convert_to_model_input(document, question, answer, tokenizer, modelConfig, device, return_tensors=True):
    """Input:
        document: a string of the document(s)
        question: a string of the question
        answer: a string of the answer
        tokenizer: if it is a string, this tokenizer will tokenize it
        modelConfig: needs to truncate input to the sequence length required (modelConfig.n_ctx)
        device: needs to put the input tensor to the device
    
    Refer to https://github.com/jojotenya/LAMOL/blob/03c31d9f0c7bf71295bc2d362ddf40a7656956e1/utils.py#L220
    
    Outputs:
        context[:args.max_len-len(example)-1] + question + ans_token + answer
        maximum of 1023 length, since the original -1 for the eos_token at the end
    """
    # Need to manually truncate it to 1024 [GPT2]
    if isinstance(document, list): # Pretokenized input, just need to convert it to tokens.
        document = tokenizer.convert_tokens_to_ids(document)
    elif isinstance(document, str): # Tokenize and encode it
        document = tokenizer.encode(document)
    else:
        raise Exception("Document should be list or string")
    question = tokenizer.encode(question)
    answer   = tokenizer.encode(answer)
    
    example = question + [tokenizer.ans_token_id] + answer
    
    if len(example) + 1 > modelConfig.n_ctx:
        logger.warning('an example with len {} is too long!'.format(len(example) + 1))
        return
    
    # -1 because there is eos_token spare for the original LAMOL
    _input = document[:modelConfig.n_ctx-len(example)-1] + example
    
    document_mask = np.zeros((len(_input)), dtype=bool)
    document_mask[:len(document[:modelConfig.n_ctx-len(example)-1])] = True
    
    # Convert to Tensors if required
    if return_tensors:
        _input = torch.tensor(_input, dtype=torch.long, device=device)
        
    return {
        'input_ids': _input,
        'document_mask': document_mask,
    }

def convert_to_tokenized_ground_truth(original_ground_truth, original_document, tokenizer):
    """ Algorithm to get new_ground_truth by the tokenizer. Checking each substring if it's equal, and appending the 
    ground_truth value of the original_document_index
    Assumptions: NO UNKNOWNS! since we check by ==, else need to check for unknowns and perform equality ignoring left side.
    
    Inputs:
        original_ground_truth: Original GT boolean array with same shape as original_document
        original_document: Original Pretokenized document array with same shape as original_ground_truth
        tokenizer: tokenizer used to encode/decode the document
        
    Output: 
        new_ground_truth: New GT boolean array expanded by tokenizer
    """
    new_document = tokenizer.encode(' '.join(original_document))
    new_ground_truth  = []
    
    original_document_start_index = 0
    original_document_end_index = 1
    new_document_start_index = 0
    new_document_end_index = 1
    
    while new_document_end_index <= len(new_document):
        original_document_temp = ' '.join(original_document[original_document_start_index:original_document_end_index])
        new_document_temp = tokenizer.decode(new_document[new_document_start_index:new_document_end_index]).strip()
        
        new_ground_truth.append(original_ground_truth[original_document_end_index-1])
        
#         if new_document_end_index < 150:
#             print("NEW DOC", new_document_temp)
#             print("ORI DOC", original_document_temp)
#             print(new_ground_truth)
        
        ## ASSUME THAT NEW_DOCUMENT_TEMP HAS NO UNKNOWNS??!?
        if new_document_temp == original_document_temp:
            original_document_start_index += 1
            original_document_end_index += 1
            new_document_start_index = new_document_end_index
        
        new_document_end_index += 1
        
    
    return new_ground_truth

def select_attention(single_attention_head):
    """Returns the aggregated results of all the tokens
    Currently just use CLS"""
#     return attention_head[0]
    # Try Averaging
    return single_attention_head.mean(axis=0)


def _auprc(true, pred):
        true = [int(t) for t in true]
        precision, recall, _ = precision_recall_curve(true, pred)
        return auc(recall, precision)

def _get_auprcs(attn_head_tuple):
    # Attn_head is Dimension [seq_len, seq_len]
    attn_head_ind, attn_head = attn_head_tuple
    sub_auprcs = [] #sub_auprcs is the auprcs from every attention head!!

#         logger.debug(f"atten head {attn_head_ind} {attn_head.shape}") #REMOVE LOGGER IN MULTIPROCESSING!!! It will not be defined

    # Attn_head_token is Dimension [seq_len], for each token compared to other tokens
    for attn_head_token_ind, attn_head_token in enumerate(attn_head):
        pred = attn_head_token
        auprc = _auprc(ground_truth,pred)

        if math.isnan(auprc):
            logger.debug(f"Attention Head Token Number {attn_head_token_ind} at Attention Head {attn_head_ind}")
            logger.debug(f"Ground_truth: {ground_truth}")
            logger.debug(f"pred: {pred}")
            logger.debug(f"auprc Detected: {auprc}")
        sub_auprcs.append(auprc)
    return sub_auprcs

def _get_ious(attn_head_tuple):
    # Attn_head is Dimension [seq_len, seq_len]
    attn_head_ind, attn_head, method, hard_selection_method, p, k, ground_truth = attn_head_tuple
    # If Ground truth has many, choose the one with attn_head_ind!
    if hasattr(ground_truth, 'shape') and len(ground_truth.shape) > 1:
        ground_truth = ground_truth[attn_head_ind]

    sub_scores = [] #sub_scores is the scores from every attention head!!

#         logger.debug(f"atten head {attn_head_ind} {attn_head.shape}") #REMOVE LOGGER IN MULTIPROCESSING!!! It will not be defined

    # Attn_head_token is Dimension [seq_len], for each token compared to other tokens
    for attn_head_token_ind, attn_head_token in enumerate(attn_head):

        # Change Prediction to Hard Selection
        if hard_selection_method == "percentile":
            pred = attn_head_token > np.percentile(attn_head_token, 100-p)
        elif  hard_selection_method == "top-k": # argsort in reverse [descending] and get the k-1 index, find all that is more 
            pred = attn_head_token >= np.argsort(attn_head_token)[::-1][k-1]

        # using iou(jaccard)/f1 (dice)
        if method=="iou-token-level":
            #score = jaccard_score(ground_truth, pred)
            # Pluem's improvement on score calculation
            score=1-pdist([np.array(pred),np.array(ground_truth)],'jaccard')
            score=score.item()
        elif method=="f1-token-level":
            score = f1_score(ground_truth, pred)

        sub_scores.append(score)
    return sub_scores

def add_arguments(_argument, *args):
    """Input:
           _argument : iterable or list  to add more static columns 
       Output:
           mapped_array: mapped array of iterable/list + static columns of args
    """
    return map(lambda x: list(x)+list(args), _argument)

def find_attn_head_max(attention_tuple):
    logger = logging.getLogger(__name__)
    # has to import here for multiprocessing to work, dont ask why.
    import numpy as np
    from scipy.spatial.distance import pdist
    """Input 
        attention block (with attention heads): Dimension  [attention_head, seq_len, seq_len]
        ground_truth/feature map              : Dimension  [seq_len] List or numpy array of [attention_head 12, seq_len]
        mask                                  : Dimension  [seq_len] 
        method                                : "auprc"/"iou"/"auprc-token-level"
        hard_selection_method                 : "top-k"/"percentile"
        k                                     : selects the top k tokens from the soft selection
        p                                     : top p percentile to choose from ie. 20 means that we use np.percentile(x, 80)
        head_level_granularity                : If true, then do head-level granularity, so returns 12 values--one for each head

    Returns 
        representative_map           : the representative map of the block and ground truth
        score_max                    : the value of the max score
    """
    attention_block, ground_truth, mask, device, method, hard_selection_method, p, k, head_level_granularity = attention_tuple

    if len(attention_block.shape) > 3:
        attention_block = attention_block.squeeze()
    
    attention_block = attention_block[:, :mask.sum(), :mask.sum()] # Since ground_truth has undefined length, may be higher
        
    if hasattr(ground_truth, 'shape') and len(ground_truth.shape) > 1:
        ground_truth = ground_truth[:, :mask.sum()]
    else:
        ground_truth    = ground_truth[:mask.sum()]  # Since ground_truth has undefined length, may be higher
        
        
        # IF THERE IS NO TRUE IN ANY PART OF THE ARRAY
        # 5Dec2020 NEED TO Remove this! Since "AA_wiki_98_26" has annotation at start_token=3854, end_token=4038, start_sentence=194, end_sentence=201
        # When we truncate at 1023, this will make it all FALSE!!! but now we use IOU, so this shouldn't be a problem?
        if not any(ground_truth):
        #    print(ground_truth)
            logger.warning("WHY ALL GROUND TRUTH IS FALSE?")
    
    

    
    # auprc default is the attention_head level, aggregated by select_attention
    if method=="auprc":
        auprcs = []
        for attn_head in attention_block:
            pred = select_attention(attn_head)

            auprc = _auprc(ground_truth,pred)
            auprcs.append(auprc)

        attn_head_max_index = np.argmax(auprcs)
        return attn_head_max_index, auprcs[attn_head_max_index]
    
    # auprc-token-level is the token level, not aggregated. for loop another level!
    # Note: auprc Fails when the input is all zeros, since then the curve will be a straight line between 0 and 1, having high area under the curve. Selection of this type of attention head means that there will be a division of zero!!!
    elif method=="auprc-token-level":
        auprcs = []
        
        pool = ProcessPool(nodes=12) # Reinstantiate this everytime we run close()
        # attention block (with attention heads): Dimension  [attention_head, seq_len, seq_len]
        res = pool.map(_get_auprcs, enumerate(attention_block))
        pool.close()
        pool.join()
        pool.clear()
        #res will get array of Dimension [attention_head] (12) with each with dimension [seq_len*seq_len]
        auprcs = [auprc for sublist in res for auprc in sublist]
        
        
        attn_head_token_max_index = np.argmax(auprcs)
        attn_head_max_index = attn_head_token_max_index // attention_block.shape[-1] # Divided by seq len to get the max attention_head 
        token_max_index     = attn_head_token_max_index % attention_block.shape[-1]  #Remainder of seq len to get token index
        logger.info(f"LEN auprc: {len(auprcs)} Argmax of AUPRC: {np.argmax(auprcs)} MAX auprc: {auprcs[attn_head_token_max_index]}")
        logger.info(f"attn_head_max_index: {attn_head_max_index} auprcs:10: {auprcs[:10]}")
        logger.info(f"attention block with head number {attn_head_max_index} and token number {token_max_index} selected.")
        logger.debug(attention_block[attn_head_max_index][token_max_index])
        logger.debug(f"REDO Auprc: {_auprc(ground_truth,attention_block[attn_head_max_index][token_max_index])}")
        return attention_block[attn_head_max_index][token_max_index], auprcs[attn_head_token_max_index]
    
    #####  IoU/Jaccard Coefficient: TP/ (TP+FP+FN)  #####
    # https://stats.stackexchange.com/questions/273537/f1-dice-score-vs-iou
    #####    F1/Dice Score: 2TP/ (2TP+FP+FN)    #####
    # Gives more importance to True positives, or in order words, penalize the FP and FN less than IoU.
    # 1 TP with 1 FP gives 2/3 while IoU gives 1/2
    elif method=="iou-token-level" or method=="f1-token-level":
        for attn_head_ind in range(12):
             # Attn_head is Dimension [seq_len, seq_len]
            # If Ground truth has many, choose the one with attn_head_ind!
            if hasattr(ground_truth, 'shape') and len(ground_truth.shape) > 1:
                ground_truth = ground_truth[attn_head_ind]
        # attention_block (with attention heads): Dimension  [attention_head, seq_len, seq_len]
        # change each attn_head_token (at dim 2) to Hard Selection
        if hard_selection_method == "percentile":
            preds = np.apply_along_axis(lambda x: x > np.percentile(x, 100-p), 2, attention_block)
        elif  hard_selection_method == "top-k": # argsort in reverse [descending] and get the k-1 index, find all that is more
            preds = np.apply_along_axis(lambda x: x >= np.argsort(x)[::-1][k-1], 2, attention_block)
        # preds (with attention heads): Dimension  [attention_head, seq_len, seq_len]
        # using iou(jaccard)/f1 (dice)
        if method=="iou-token-level":
            ### this is iou but way faster ###
            scores = np.apply_along_axis(lambda x: (1-pdist([np.array(x),np.array(ground_truth)],'jaccard')).item(), 2, preds)
        elif method=="f1-token-level":    	
            scores = np.apply_along_axis(lambda x: f1_score(ground_truth, pred), 2, preds)
        
        if not head_level_granularity: 
            attn_head_token_max_index = np.argmax(scores) #flatten argmax!
            attn_head_max_index, token_max_index = np.unravel_index(attn_head_token_max_index, scores.shape) #unravel flatten to tuple (i,j)

            logger.info(f"LEN scores: {len(scores)} Argmax of scores: {np.argmax(scores)} MAX score: {scores[attn_head_max_index, token_max_index]}")
            logger.info(f"attn_head_max_index: {attn_head_max_index} auprcs:10: {scores[:10]}")
            logger.info(f"attention block with head number {attn_head_max_index} and token number {token_max_index} selected.")
            return attention_block[attn_head_max_index][token_max_index], scores[attn_head_max_index, token_max_index]
        else:
            attn_head_token_max_indices = np.argmax(scores,axis=1) # Will be shape (12) ie. [771 771 ... 288 770 746 773 773 772 255]
            
            logger.info(f"attn_head_token_max_indices: {attn_head_token_max_indices}")
            logger.info(f"scores: {scores[np.arange(12), attn_head_token_max_indices]}")
            # Will Return rm_mo_gt of shape [12, seq_len] and scores of shape [12] 
            return attention_block[np.arange(12), attn_head_token_max_indices], scores[np.arange(12), attn_head_token_max_indices]

    
if __name__ =="__main__": 
    
    parser = argparse.ArgumentParser(description="Runing Critical Freezing Algorithm")
    parser.add_argument("--head_level", help="Do head level Granularity",action="store_true")
    parser.add_argument("--head_level_top_k", help="Number of Heads to choose from", type=int, default=12)
    parser.add_argument("--data_dir", help="The data to put in to the algorithm", type=str, choices=['movies', 'scifact', 'boolq'], required=True)
    parser.add_argument("--old_model_dir", help="The folder of the old model", type=str, default="./bms_M1M2/task1")
    parser.add_argument("--new_model_dir", help="The folder of the new model", type=str, default="./bms_M1M2/task2")
    parser.add_argument("--mo_gt_method", help="Method to select from Model Old to Ground Truth", 
                        type=str, default="iou-token-level", choices=['iou-token-level',])
    parser.add_argument("--mn_mo_method", help="Method to select from Model New to Model Old", 
                        type=str, default="iou-token-level", choices=['iou-token-level',])
    parser.add_argument("--device", help="Device to use 'cpu' or 'cuda:0'/'cuda:1'", 
                        type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument("-n", "--n_ann", help="Number of maximum annotations to do ie. 100", 
                        type=int, default=10000)
    parser.add_argument("--gen_rat", help="Use generated rationale?", action="store_true")
    args = parser.parse_args()
    
    HEAD_LEVEL_GRANULARITY = args.head_level # If False, then do block level granularity
    HEAD_LEVEL_TOP_K       = args.head_level_top_k   # Number of Heads to choose from
    MO_GT_METHOD = args.mo_gt_method
    MN_MO_METHOD = args.mn_mo_method
    MAX_NO_ANNOTATIONS = args.n_ann
    
    data_root = os.path.join('data', args.data_dir)
    
    OLD_MODEL_DIR = Path(args.old_model_dir)
    OLD_TOK_DIR = OLD_MODEL_DIR

    NEW_MODEL_DIR = Path(args.new_model_dir)
    NEW_TOK_DIR = NEW_MODEL_DIR
    
    device = torch.device(args.device)
    
    hard_selection_method="percentile"
    k=100
    p=20
    
    
    # datetime object containing current date and time
    now = datetime.now()
    LOG_FILE = f"{now.strftime('%Y-%m-%dT%H.%M.%S')}-{args.old_model_dir.split('/')[1]}-{'head' if args.head_level else 'block'}-{args.device[:4]}-n{args.n_ann}.log"
    
    logging.basicConfig(filename=LOG_FILE)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # logger.setLevel(logging.INFO)
    # logger.setLevel(logging.WARN)
    
    print(f"Logging at {LOG_FILE}...")
    print(f"Loading data from {data_root}")
    print(f"Old Model at {OLD_MODEL_DIR}, New Model at {NEW_MODEL_DIR}")
    print(f"Using Device {args.device}")
    print(f"Beginning with HEAD_LEVEL_GRANULARITY {HEAD_LEVEL_GRANULARITY} HEAD_LEVEL_TOP_K {HEAD_LEVEL_TOP_K}")
    print(f"MO_GT_METHOD {MO_GT_METHOD} MN_MO_METHOD {MN_MO_METHOD}")
    print(f"MAX_NO_ANNOTATIONS {MAX_NO_ANNOTATIONS}")
    
    
    ############################
    ## Start Importing Models ##
    ############################
    print("Importing old and new models...")
    tic = time.time()
    ## Import Old Model 
    model_old_config = GPT2Config.from_json_file(OLD_MODEL_DIR/"config.json")
    model_old_config.output_attentions = True
    model_old = GPT2LMHeadModel(model_old_config).to(device)
    model_old.load_state_dict(torch.load(OLD_MODEL_DIR/"model-5", map_location=device))

    ## Import New Model 
    model_new_config = GPT2Config.from_json_file(NEW_MODEL_DIR/"config.json")
    model_new_config.output_attentions = True
    model_new = GPT2LMHeadModel(model_new_config).to(device)
    model_new.load_state_dict(torch.load(NEW_MODEL_DIR/"model-5", map_location=device))

    model_old.to(device)
    model_new.to(device)
    print(f"Ended importing models in {time.time()-tic}s")
    ############################
    ## End Importing Models   ##
    ############################
    
    
    
    ##########################
    ## Start Get Tokens Map ##
    ##########################
    print("Starting get tokens map...")
    tic = time.time()
    # From LAMOL/settings.py
    # special_tokens = {"ans_token":'__ans__', "pad_token":'__pad__', "unk_token":'__unk__', "eos_token": '<|endoftext|>'}
    # tokenizer.add_tokens(list(special_tokens.values()))

    with open(NEW_TOK_DIR/"special_tokens_map.json") as f:
        special_tokens_map = json.load(f)
    print(f"special_tokens_map: {special_tokens_map}")

    with open(NEW_TOK_DIR/"added_tokens.json") as f:
        added_tokens = json.load(f)
    print(f"added_tokens: {added_tokens}")


    tokenizer = GPT2Tokenizer(NEW_TOK_DIR/"vocab.json", NEW_TOK_DIR/"merges.txt")
    tokenizer.add_tokens(list(added_tokens.keys()))
    # print(token)
    print(f"Total # of tokens: {len(tokenizer)}")


    tokenizer.ans_token = "__ans__"
    tokenizer.ans_token_id = tokenizer.convert_tokens_to_ids("__ans__")


    for k,v in special_tokens_map.items():
        assert tokenizer.special_tokens_map[k] == v
    for tok, tok_id in added_tokens.items():
        assert tokenizer.convert_ids_to_tokens(tok_id) == tok
    print(f"<special_tokens_map and added_tokens matched successfully> in {time.time()-tic}s")
    ##########################
    ## End Get Tokens Map ##
    ##########################
    
    
    ####################
    ## Start Get Data ##
    ####################
    print("Starting get data...")
    tic = time.time()
    # annotations is the list of all annotations in val.jsonl
    if not args.gen_rat:
        annotations = annotations_from_jsonl(os.path.join(data_root, 'val.jsonl'))
    else:
        print('USING GENERATED RATIONALE')
        annotations = annotations_from_jsonl(os.path.join(data_root, 'val_gen.jsonl'))
    # docids is the list of all document ids (note: one annotation may have many docids)
    docids = sorted(set(chain.from_iterable((ev.docid for ev in chain.from_iterable(ann.evidences)) for ann in annotations)))
    # flattened_documents is a dictionary from key {docid} -> [list of tokens t1, t2, t3]
    flattened_documents = load_flattened_documents(data_root, docids)

    # key_to_model_input is a dictionary from {annotation_id} -> {model_input} for that particular annotation
    # key_to_annotation is a dictionary from {annotation_id} -> GT for that particular annotation (tokenized)
    key_to_model_input = dict()
    key_to_annotation = dict()
    # _key_to_original_gt is an intermediate product temporary original ground truth dictionary map {(annotation_id, docid)} -> original GT (word-based tokens)
    _key_to_original_gt = dict()
    
    # For every evidence in the evidence list of the annotation:
    #   1. Find model_input 
    #   2. Find annotation
    for ann in tqdm(annotations[:MAX_NO_ANNOTATIONS]):

        # Find the set of unique docids for that particular annotation
        _ann_docids = tuple(sorted(set(ev.docid for ev in chain.from_iterable(ann.evidences))))
        
        # All documents' tokens extended together
        _flattened_docs = functools.reduce(
                             lambda flattened_token_list, new_docid : flattened_token_list + flattened_documents[new_docid],
                             _ann_docids[1:],
                             flattened_documents[_ann_docids[0]]
                           )
        

        ### 1. Convert Document, Question, Answer to model input ###
        # Join all the tokens of all documents in the docid, and tokenize with tokenizer
        # Note: Needs initializer because it will breakdown when there is only 1 docid. so ONLY reduce if there are more than 2!!!
        _input = convert_to_model_input(' '.join(_flattened_docs),
                                       ann.query,  
                                       ann.classification,
                                       tokenizer,
                                       model_new_config,
                                       device)
        ### add to annotation_id -> _input
        key_to_model_input[ann.annotation_id] = _input

        
        
        ### 2. Find all evidences and convert to ground truth  ###
        
        # 2.1 Create temporary original ground truth dictionary map _key_to_original_gt {(annotation_id, docid)} -> original GT
        #      mark True for every start_token and end_token
        #      ann.annotation_id, ev.docid is NOT the same for boolq and scifact, only true for movies dataset
        #      1 annotation_id may refer to MULTIPLE docids!!!
        for ev in chain.from_iterable(ann.evidences):
            key = (ann.annotation_id, ev.docid)

            if key not in _key_to_original_gt:
                _key_to_original_gt[key] = [False for _ in flattened_documents[ev.docid]]

            start, end = ev.start_token, ev.end_token

            for t in range(start, end):
                _key_to_original_gt[key][t] = True
        # End 2.1 #
        
        # 2.2 Convert all _key_to_original_gt to CONCAT-ed tokenized GT in key_to_annotation
        tokenized_ground_truth = functools.reduce(
                             lambda flattened_token_list, new_docid : flattened_token_list + \
                                 convert_to_tokenized_ground_truth( 
                                     _key_to_original_gt[(ann.annotation_id, new_docid)],
                                     flattened_documents[new_docid],
                                     tokenizer
                                 ),
                             _ann_docids[1:],
                             convert_to_tokenized_ground_truth(
                                 _key_to_original_gt[(ann.annotation_id, _ann_docids[0])],
                                 flattened_documents[_ann_docids[0]],
                                 tokenizer
                             )
                           )
        key_to_annotation[ann.annotation_id] = tokenized_ground_truth
        # End 2.2 #
    
    print(f"Ended get data in {time.time()-tic}s")
    ####################
    ## End Get Data ##
    ####################


    #####################
    ## Start Algorithm ##
    #####################
    block_L = []
    
    ### Time Log Definitions ###
    time_convert_model_log = []
    time_predict_model_old_log = []
    time_predict_model_new_log = []
    time_find_attnhead_max_gt_log = []
    time_find_top20_log = []
    time_find_attnhead_max_new_log = []
    time_global = time.time()

    for ann in tqdm(annotations[:MAX_NO_ANNOTATIONS]):
        logger.info(f"Document IDs: {tuple(sorted(set(ev.docid for ev in chain.from_iterable(ann.evidences))))}")
        logger.info(f"Document: {key_to_model_input[ann.annotation_id]['input_ids'][:200]}")
        logger.info(f"Question: {ann.query}")
        logger.info(f"Answer: {ann.classification}")

        ### 1. Convert Document, Question, Answer to model input ###
        tic_convert_model_log = time.time()
        
        _input = key_to_model_input[ann.annotation_id]

        input_ids = _input['input_ids']
        document_mask = _input['document_mask']
        ground_truth = key_to_annotation[ann.annotation_id]

        input_ids = input_ids.reshape([1, -1])

        logger.info(f"Input Shape: {input_ids.shape}")
        logger.debug(tokenizer.decode(input_ids.squeeze().tolist()))
        logger.info(f"Document Mask Sum: {document_mask.sum()}")
        
        time_convert_model_log.append(time.time()-tic_convert_model_log)

        ### 2. Predict the attentions from the input tokens ###
        tic_predict_model_old_log = time.time()
        last_hidden_state_old, pooler_output_old, attentions_old = model_old(input_ids)
        logger.info(f"Attention Blocks: {len(attentions_old)} First attention block old shape: {attentions_old[0].shape}")
        time_predict_model_old_log.append(time.time()-tic_predict_model_old_log)
        
        tic_predict_model_new_log = time.time()
        last_hidden_state_new, pooler_output_new, attentions_new = model_new(input_ids)
        logger.info(f"Attention Blocks: {len(attentions_new)} First attention block new shape: {attentions_new[0].shape}")
        time_predict_model_new_log.append(time.time()-tic_predict_model_new_log)
        
        
        
         # Pluem: detach here seems to make it faster, not sure tho
        if device.type == "cuda":
            attentions_old = [attn_old.cpu().detach() for attn_old in attentions_old]
            attentions_new = [attn_new.cpu().detach() for attn_new in attentions_new]
        else:
            attentions_old = [attn_old.detach() for attn_old in attentions_old]
            attentions_new = [attn_new.detach() for attn_new in attentions_new]
        
        # attentions is a list of attention blocks (12), 
        #   where each attention has the dimension [batch_size, attention_head, seq_len, seq_len]
        
        
        ### find_attn_head_max for attentions_old (all 12 blocks) ###
        # block first dimension is batchsize! - need to squeeze it out since it's always (1)
        # Block has dimension [batch_size, attention_head, seq_len, seq_len] where batch_size=1
#         block_old = block_old.squeeze() # Dimension  [attention_head, seq_len, seq_len]
#         block_new = block_new.squeeze() # Dimension [attention_head, seq_len, seq_len]
        logger.debug(f"==== STARTING Finding Attention Head Max to GT ====" )
        tic_find_attnhead_max_gt_log = time.time()
        pool = ProcessPool(nodes=12)
        
        out = pool.map(find_attn_head_max, add_arguments(attentions_old, ground_truth, document_mask, device, MO_GT_METHOD, hard_selection_method, p, k, HEAD_LEVEL_GRANULARITY ))
        # out shape is [no_of_block, [rm_mo_gt,max_mo_gt]]
        pool.close()
        pool.join()
        pool.clear()
        time_find_attnhead_max_gt_log.append(time.time()-tic_find_attnhead_max_gt_log)
        
        
        tic_find_top20_log = time.time()
        rm_mo_gts = [rm_mo_gt for rm_mo_gt,max_mo_gt in out]
        max_mo_gts = [max_mo_gt for rm_mo_gt,max_mo_gt in out]
        
        for rm_mo_gt in rm_mo_gts:
            logger.debug(f"==== STARTING Finding Top 20 Percentile ====" )
            # Change rm_mo_gt Representative Map of Old model and Ground Truth -> Boolean Array for top 20 percentile
            if not HEAD_LEVEL_GRANULARITY: # Handle rm_mo_gt with shape [seq_len]
                rm_mo_gt_top20 = rm_mo_gt > np.percentile(rm_mo_gt, 80)
            else: # Handle rm_mo_gt with shape [12,seq_len]
                # Need to expand and transpose to vertically stack the percentiles 
                #   ie. [[8.99920531e-04], [1.10337669e-05], ... [3.12965992e-03]] -> groundtruth of dimension [12, seq_len]
                rm_mo_gt_top20 = rm_mo_gt.numpy() >  np.expand_dims(np.percentile(rm_mo_gt, 80, axis=1), axis=0).T
            logger.debug(f"rm_mo_gt {rm_mo_gt}")
            logger.debug(f"rm_mo_gt_top20 {rm_mo_gt_top20}")
        time_find_top20_log.append(time.time()-tic_find_top20_log)
            
        tic_find_attnhead_max_new_log = time.time()
        pool = ProcessPool(nodes=12)
        ##find_attn_head_max for attentions_new (all 12 blocks)
        out = pool.map(find_attn_head_max, add_arguments(attentions_new, rm_mo_gt_top20, document_mask, device, MN_MO_METHOD, hard_selection_method, p, k, HEAD_LEVEL_GRANULARITY))
        # out shape is [no_of_block, [rm_mn_mo,max_mn_mo]]
        pool.close()
        pool.join()
        pool.clear()
        time_find_attnhead_max_new_log.append(time.time()-tic_find_attnhead_max_new_log)
        
        rm_mn_mos = [rm_mn_mo for rm_mn_mo,max_mn_mo in out] 
        max_mn_mos = [max_mn_mo for rm_mn_mo,max_mn_mo in out]
        
        
        block_scores = max_mn_mos # List of max IOU MO-MN
        block_rm = rm_mn_mos    # List of representative maps of MO-MN, dunno what to do with it
        
        del out
        del max_mn_mos
        del rm_mn_mos


            



        # Block with highest drop in IOU
        if not HEAD_LEVEL_GRANULARITY:
            b = np.argmin(block_scores)
            block_L.append(b)
#             print(block_L)
        else:
            # block_scores is now [12 blocks, 12 attention heads] array
            block_scores = np.vstack(block_scores)
            top_indices = np.argsort(block_scores, axis=None)[:HEAD_LEVEL_TOP_K] # argsort on flattened array, and find TOP_K MINIMUM
            block_indices, atn_head_indices = np.unravel_index(top_indices, block_scores.shape)
            b = list(zip(block_indices, atn_head_indices))
            block_L.extend(b) # Extend because b is an array of #HEAD_LEVEL_TOP_K of tuples of (block_index, atn_head_index)
#             print(block_L)


        ## ADD BREAK FOR 1 DOCUMENT
#         break
    # Most frequent block in block_L
    if not HEAD_LEVEL_GRANULARITY:
        print("Most frequent block:" ,stats.mode(block_L))

    cnt = Counter()
    for block in block_L:
        cnt[block] += 1
    print("Total Counter")
    print(cnt)
    print("Most Common 12")
    print(cnt.most_common(12))
    
    
    ## Write all times!
#     with open("time_log/global.csv", 'a') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow([LOG_FILE, time.time() - time_global])
#     with open("time_log/most_common.csv", 'a') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow([LOG_FILE] + cnt.most_common(1000))
#     with open("time_log/1convert_model.csv", 'a') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow([LOG_FILE] + time_convert_model_log)
#     with open("time_log/2predict_model_old.csv", 'a') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow([LOG_FILE] + time_predict_model_old_log)
#     with open("time_log/3predict_model_new.csv", 'a') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow([LOG_FILE] + time_predict_model_new_log)
#     with open("time_log/4find_attnhead_maxgt.csv", 'a') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow([LOG_FILE] + time_find_attnhead_max_gt_log)
#     with open("time_log/5find_top20.csv", 'a') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow([LOG_FILE] + time_find_top20_log)
#     with open("time_log/6find_attnhead_maxnew.csv", 'a') as f:
#         csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#         csv_writer.writerow([LOG_FILE] + time_find_attnhead_max_new_log)
