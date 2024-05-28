# Unlocking Emergent Modularity in Large Language Models

## Repo Overview

For more details about the method, please take a look at our [paper](https://arxiv.org/abs/2310.10908).

In this section, we provide an overview of the three main components of the repository.

**LLMs**: This section includes experiments tunning Llama on Alpaca. We use [opencompass](https://github.com/open-compass/OpenCompass/) for evaluation.

**Vision**: This section includes experiments involving full fine-tuning on Domainbed. It focuses on vision-related tasks and evaluations.

**Language**: This section contains experiments related to full fine-tuning and LoRA tuning on the GLUE dataset. Additionally, experiments on ID GLUE and OOD GLUE-X are conducted. These experiments primarily pertain to language-related tasks.

Demons: This section provides the data sources and plotting code for all the images featured in the original paper.

Tutel: Modified from [Tutel MoE](https://github.com/microsoft/tutel) to support the added avg-k gaing.

For specific instructions on running the code for each component, please refer to the README.md file within the corresponding folder.

## Environments

Please prepare the Vision and Language environments separately and follow the instructions in each part. **Importantly**, we add functions like avg-k gating to the original [Tutel MoE](https://github.com/microsoft/tutel), so please install tutel from the local file `tutel` through cd `./tutel` and `pip install ./`.


## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{qiu2023emergent,
  title={Emergent Mixture-of-Experts: Can Dense Pre-trained Transformers Benefit from Emergent Modular Structures?},
  author={Qiu, Zihan and Huang, Zeyu and Fu, Jie},
  journal={arXiv preprint arXiv:2310.10908},
  year={2023}
}
```

## Acknowledgement

The MoE module is built on [Tutel MoE](https://github.com/microsoft/tutel). Notice we have added the avg-k function to the original gate, so please install it from the local file `tutel` and follow the corresponding instructions.

The Vision codebase is built on [GMoE](https://github.com/Luodian/Generalizable-Mixture-of-Experts) and original [Domainbed](https://github.com/facebookresearch/DomainBed)

The language training module is built on [Transformers](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification), OOD evaluation is built on [GLUE-X](https://github.com/YangLinyi/GLUE-X).

The MoE split method is built on [MoEfication](https://github.com/thunlp/MoEfication)

## License

This source code is released under the MIT license, included [here](LICENSE).
