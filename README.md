# SNCSE
SNCSE: Contrastive Learning for Unsupervised Sentence Embedding with Soft Negative Samples

SNCSE aims to alleviate feature suppression in contrastive learning for unsupervised sentence embedding. In the field, feature suppression means the models fail to distinguish and decouple textual similarity and semantic similarity. As a result, they may overestimate the semantic similarity of any pairs with similar textual regardless of the actual semantic difference between them. And the models may underestimate the semantic similarity of pairs with less words in common. (Please refer to Section 5 of our paper for several instances and detailed analysis.) To this end, we propose to take the negation of original sentences as soft negative samples, and introduve them into the traditional contrastive learning framework through bidirectional margin loss (BML). The structure of SNCSE is as follows:

![models2](https://user-images.githubusercontent.com/49329979/149649193-849afb0a-6cdf-4944-90ff-eb917ef8653a.png)

The performance of SNCSE on STS task with different encoders is:

![image](https://user-images.githubusercontent.com/49329979/149649862-f33ef789-af2f-495f-b52c-f2336d9ba3f5.png)

To reproduct above results, please [download] the models, adjust the file path variables and run:

python bert_prediction.py
python roberta_prediction.py

To 
Feel free to contact the authors at wanghao2@sensetime.com for any questions.
