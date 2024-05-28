# reconstruct the weight of the original llama model
# 对weight进行聚类并且处理交换
from k_means_constrained import KMeansConstrained
import torch
import os

def cluster_weights(
    weight, 
    weight_name,
    n_clusters, 
    cache_dir=None
):
    """
    weight: torch.tensor
    n_clusters: int
    axis: int
    layer: int
    cache_dir: str
    """
    shape = weight.shape # intermediate_size, embed_size
    expert_size = shape[0] // n_clusters
    index_name = f"{weight_name}_clusters{n_clusters}_shape{shape}.pth"
    if cache_dir is not None and os.path.exists(os.path.join(cache_dir, index_name)):
        index = torch.load(os.path.join(cache_dir, index_name))
    else: 
        weight = torch.nn.functional.normalize(weight, p=2, dim=-1)
        weight = weight.detach().cpu().numpy()
        kmeans = KMeansConstrained(
            n_clusters=n_clusters, size_min=expert_size,
            size_max=expert_size, random_state=0, n_jobs=16,
            max_iter=1000)
        kmeans.fit(weight)
        index = torch.from_numpy(kmeans.labels_)
        if cache_dir is not None:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            torch.save(index, os.path.join(cache_dir, index_name))

    return index


def rearrange_weight(
    weight, 
    index,
    n_expert,
    
):
    dim1, dim2 = weight.shape
    # rearrange the weight
    if dim1 > dim2:
        expert_size = dim1 // n_expert
        # we are rearranging the gate_proj and up _proj
        new_weight = torch.zeros_like(weight)
        for i in range(n_expert):
            new_weight[i * expert_size: (i+1) * expert_size] = weight[index == i]
        return new_weight
    else:
        expert_size = dim2 // n_expert
        # we are processing the down_proj
        new_weight = torch.zeros_like(weight.T)
        for i in range(n_expert):
            new_weight[i * expert_size: (i+1) * expert_size] = weight.T[index == i]
        return new_weight.T



# 运行 main
if __name__== '__main__':
    import argparse

    from transformers import LlamaForCausalLM, LlamaTokenizer
    from collections import OrderedDict

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_experts', type=int, default=64)
    parser.add_argument('--source_dir', type=str, default='/home/hkustadmin/huangzeyu/Llama-2-7b-hf')
    parser.add_argument('--output_dir', type=str, default='/home/hkustadmin/huangzeyu/Llama-2-7b-hf-moe')
    parser.add_argument('--cache_dir', type=str, default='/home/hkustadmin/huangzeyu/MoELlama/moe_clusters')
    args = parser.parse_args()
    
    args.output_dir =args.output_dir + f"-{args.n_experts}"

    model = LlamaForCausalLM.from_pretrained(args.source_dir)
    tokenizer = LlamaTokenizer.from_pretrained(args.source_dir)
    sd = OrderedDict()
    ori_sd = model.state_dict()
    print("begin processing weighs!")
    for n, p in ori_sd.items():
        if 'mlp' not in n:
            sd[n] = p
            continue
        if n in sd:
            continue
        
        layer = int(n.split('.')[2])

        pfx = '.'.join(n.split('.')[:4])
        print(f"processing {pfx}")
        gate_prog_weight = ori_sd[f'{pfx}.gate_proj.weight']
        indice = cluster_weights(
            weight=gate_prog_weight,
            weight_name=pfx,
            n_clusters=args.n_experts,
            cache_dir=args.cache_dir
        )
        sd[f'{pfx}.gate_proj.weight'] = rearrange_weight(
            gate_prog_weight, indice, args.n_experts)
        sd[f'{pfx}.up_proj.weight'] = rearrange_weight(
            ori_sd[f'{pfx}.up_proj.weight'], indice, args.n_experts)
        sd[f'{pfx}.down_proj.weight'] = rearrange_weight(
            ori_sd[f'{pfx}.down_proj.weight'], indice, args.n_experts)
    
    model.load_state_dict(sd)
   
    print("begin saving model!")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("finished!")
    