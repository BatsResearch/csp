# Credits
​
We particularly appreciate the annotation efforts from [Andrew Delworth](https://www.linkedin.com/in/andy-delworth-2a73b31a9) and [Elise Carman](https://www.linkedin.com/in/elise-carman-9914b6154/) for the attribute-attribute object dataset. This project is built on top of the ideas of CLIP, compositional zero-shot learning and language model prompting. 
​

## CLIP
[https://github.com/openai/CLIP](https://github.com/openai/CLIP).
```
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever. (2021). Learning Transferable Visual Models From Natural Language Supervision.
```

​
## Datasets
We obtained the seen/unseen split information for the datasets with the `download_data.sh` scripts supplied by [https://github.com/ExplainableML/czsl](https://github.com/ExplainableML/czsl)
For evaluation, we used the following datasets.
### MIT-States
[http://web.mit.edu/phillipi/Public/states_and_transformations/index.html](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html).
```
Phillip Isola*, Joseph J. Lim*, and Edward H. Adelson. (2015). Discovering States and Transformations in Image Collections. 
```
### UT-Zappos
[https://vision.cs.utexas.edu/projects/finegrained/utzap50k/](https://vision.cs.utexas.edu/projects/finegrained/utzap50k/).
```
A. Yu and K. Grauman. (2014). Fine-Grained Visual Comparisons with Local Learning.
```
### C-GQA
[https://arxiv.org/pdf/2102.01987.pdf](https://arxiv.org/pdf/2102.01987.pdf).
​
```
Muhammad Ferjad Naeem, Yongqin Xian, Federico Tombari, Zeynep Akata (2021). Learning Graph Embeddings for Compositional Zero-shot Learning.
```

## Code
The evaluation for compositional zero-shot learning is based on the following codebases:
​
### CZSL 
We obtained the code from [https://github.com/ExplainableML/czsl](https://github.com/ExplainableML/czsl).
​
### ProtoProp 
We obtained the code from [https://github.com/FrankRuis/protoprop](https://github.com/FrankRuis/protoprop)