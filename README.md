# ABSA_GR
----------------------------------------------------
## MSc Data & Web Science, Aristotle University of Thessaloniki (AUTH)
#### Project: *“Aspect-Based Sentiment Analysis”*

----------------------------------------------------
**Abstract**:
Aspect-Based Sentiment Analysis is a specific category of classic Sentiment Analysis that focuses on identifying the sentiment towards each individual aspect given in a document. As an aspect, we usually refer to the component of an object, that is addressed either implicitly or explicitly in a text. Highly used by businesses worldwide, especially marketing departments, Aspect-Based Sentiment Analysis has seen a sudden rise due to advances in Deep Learning. Businesses resort to Aspect-Based Sentiment Analysis techniques to gather information regarding the appeal of their products, aiming at increased profits in the longer run. Most techniques nowadays, utilize transformer models due to the advances in that area. This thesis presents a novel Greek annotated dataset for Aspect-Based Sentiment Analysis, that is comprised by a set of reviews derived from a variety of social media platforms, including Twitter, Facebook and YouTube. Additionally, we propose a series of BERT architectures for Greek, a generally low-resource language.  Utilizing the GreekBERT model we offer different solutions that solve this issue, outperforming all other baseline models presented here.

----------------------------------------------------
**Data Format:**
| ID | CreateDate | Channel | Text | Aspect | Sentiment |
|     :---:      |     :---:      |     :---:      |     :---:      |     :---:      |     :---:      |
| String   | Date   | String   | String   | String     | String    |


----------------------------------------------------
**Code Structure:**
```
.
└── ABSA_GR
    ├── bert_arrays
    │   └── (stores .npy arrays)
    ├── data
    │   └── (initial set of data)
    ├── data_custom
    │   └── (results of preprocess - STEP 01)
    ├── fasttext
    │   └── (FastText embeddings .bin file)
    ├── final_datasets
    │   └── (results of preprocess - STEP 02)
    ├── models
    │   ├── bert_models.py
    │   ├── bert_rnn_models.py
    │   ├── rnn_models.py
    │   ├── rnn_models_list.py
    │   └── scikit_models.py
    ├── preprocessing
    │   ├── create_datasets.py
    │   ├── imbalance.py
    │   ├── initiate_preprocess_steps.py
    │   ├── nlp_preprocessing.py
    │   ├── preprocess.py
    │   └── vectorize.py
    ├── reading_data
    │   ├── read.py
    │   └── utils.py
    ├── README.md
    └── main.py
```
----------------------------------------------------
**GreekBERT C4L Architecure:**

![model_BERTC4H](https://user-images.githubusercontent.com/48099515/155877497-dcfbf706-3382-4ab7-84f5-906f2d2f31aa.png)

----------------------------------------------------
**F1-Macro Results:**
| | Original Dataset | Oversampled Dataset | Hybrid Dataset |
|     :---      |     :---:      |     :---:      |     :---:      |
| GreekBERT 3DL  | 51.73%   | 71.52%     | 72.79%    |
| GreekBERT GRU  | 52.61%   | 72.13%     | 73.29%    |
| GreekBERT S4L  | 53.00%   | 71.39%     | 74.38%    |
| GreekBERT C4L  | 54.23%   | 71.67%     | 74.41%    |
