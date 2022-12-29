## Beyond Digital “Echo Chambers”: The Role of Viewpoint Diversity in Political Discussion

Code, sample data, and other supplementary material for the paper "Beyond Digital “Echo Chambers”: The Role of Viewpoint Diversity in Political Discussion" accepted at WSDM'23

The paper can be found here: [Beyond Digital “Echo Chambers”: The Role of Viewpoint Diversity in Political Discussion](https://arxiv.org/abs/2212.09056)

Our online talk for WSDM 2023 will be posted soon.

If you use our work, please cite us:

```
@article{hada2022beyond,
  title={Beyond Digital" Echo Chambers": The Role of Viewpoint Diversity in Political Discussion},
  author={Hada, Rishav and Fard, Amir Ebrahimi and Shugars, Sarah and Bianchi, Federico and Rossini, Patricia and Hovy, Dirk and Tromble, Rebekah and Tintarev, Nava},
  journal={arXiv preprint arXiv:2212.09056},
  year={2022}
}
```

Each folder in this repository contains separate readme with instructions.

fragmentation_computation.py: code to compute fragmentation values. Takes conversation network constructed in “conversation_retrieval/3_conversation_reconstruction.py” as input.

```jsx
python fragmentation_computaion.py
```

representation.py: code to compute representation values. Takes a list of conversations as input. Each conversation in the list is a list of labels per tweet. Ex. [[L1,L2,L2,L4],…..,[L4,L3,L1,L1,L2]].

```jsx
python representation.py
```

dyadic_interaction.py: code to compute dyadic interaction values.