# 🌍 Multilingual Neural Machine Translation Using Transformer

A neural machine translation project built using **TensorFlow/Keras** and a custom **Transformer architecture** to translate sentences from **Portuguese, Spanish, and French into English**.

## 📌 Project Overview

The goal of this project is to build a machine translation system using parallel datasets containing sentences in different languages and their corresponding English translations.

Three separate translation pipelines are implemented:

* 🇵🇹 Portuguese → 🇬🇧 English
* 🇪🇸 Spanish → 🇬🇧 English
* 🇫🇷 French → 🇬🇧 English

The project implements the Transformer architecture from scratch using TensorFlow/Keras, including positional encoding, encoder layers, decoder layers, attention mechanisms, masking, and a final vocabulary projection layer.

---

## 🎯 Objectives

The main objectives of this project are:

* Prepare and preprocess multilingual translation datasets.
* Tokenize source and target sentences.
* Convert sentences into numerical token sequences.
* Implement a Transformer architecture using TensorFlow/Keras.
* Train the model first on a small dataset to verify learning.
* Train larger Transformer models on the available translation data.
* Build a translator for generating English translations.
* Test the models using sentences in Portuguese, Spanish, and French.

---

## 🗂️ Dataset

The project uses three language-specific datasets together with an English dataset.

| Dataset    | Language Pair        |
| ---------- | -------------------- |
| `df_en_pt` | Portuguese → English |
| `df_en_es` | Spanish → English    |
| `df_en_fr` | French → English     |

The datasets are joined using the `talk_id` column, with the corresponding `transcript` columns used as the source and target text.
The notebook uses the following files:

```text
ted_talks_en.csv
ted_talks_pt-br.csv
ted_talks_es.csv
ted_talks_fr.csv
```

> **Note:** The dataset files are not included in this repository if they exceed GitHub's file-size limits. Download the required datasets separately and place them in the project directory.

---

## 🧠 Model Architecture

The project implements a custom Transformer consisting of:

```text
Input Sentence
      │
      ▼
Tokenization
      │
      ▼
Embedding
      │
      ▼
Positional Encoding
      │
      ▼
┌─────────────────┐
│     Encoder     │
│                 │
│ Multi-Head      │
│ Self Attention  │
│       ↓         │
│ Feed Forward    │
└─────────────────┘
      │
      ▼
Encoder Output
      │
      ▼
┌─────────────────┐
│     Decoder     │
│                 │
│ Masked Self     │
│ Attention       │
│       ↓         │
│ Cross Attention │
│       ↓         │
│ Feed Forward    │
└─────────────────┘
      │
      ▼
Output Projection
      │
      ▼
English Translation
```

The Transformer contains:

* Positional Encoding
* Encoder
* Decoder
* Multi-Head Attention
* Feed-Forward Networks
* Layer Normalization
* Dropout
* Padding masks
* Look-ahead masks
* Vocabulary projection layer

---

## ⚙️ Model Configuration

The notebook defines separate configurations for a small demonstration model and a larger training model.

### Tiny Model

```text
Layers       : 1
d_model      : 32
Attention heads: 2
dff          : 64
Samples      : 500
Epochs       : 30
```

### Full Model

```text
Layers       : 2
d_model      : 128
Attention heads: 4
dff          : 256
Samples      : 10,000
Epochs       : 30
```

The shared vocabulary size is **8,000 tokens**, with a maximum source sequence length of **40 tokens**.

---

## 🔤 Tokenization

TensorFlow/Keras `TextVectorization` is used to convert sentences into integer token sequences.

The target English sentences are augmented with special start and end tokens:

```text
[START] sentence [END]
```

The configured vocabulary size is 8,000 tokens.

Example tokenization dimensions from the notebook include:

```text
Portuguese tokens: (3975, 40)
English tokens:    (3975, 42)

Spanish tokens:    (3915, 40)
English tokens:    (3915, 42)

French tokens:     (3860, 40)
English tokens:    (3860, 42)
```

---

## 🏋️ Training

The model uses:

* **Optimizer:** Adam
* **Learning rate:** `1e-3`
* **Loss:** Sparse Categorical Crossentropy
* **Batch size:** `64`
* **Teacher forcing:** Yes

During training, the target sequence is divided into decoder input and target output:

```python
tar_inp = tar[:, :-1]
tar_real = tar[:, 1:]
```

The model then predicts the next token in the English sequence.

---

## 📈 Training Result

The tiny French → English model was trained for 30 epochs.

The training loss decreased from:

```text
Epoch 1  → 8.8694
Epoch 30 → 5.2248
```

This demonstrates that the model was learning from the training data, although the generated translations show that the current model still needs improvement for high-quality translation.

---

## 🧪 Example Predictions

The notebook tests the trained models using example sentences.

### Spanish → English

Example input:

```text
Este es mi primer día.
```

Expected translation:

```text
This is my first day.
```

The notebook generates an English sequence from the Spanish input.

### French → English

Example input:

```text
C'est mon premier jour.
```

Expected translation:

```text
This is my first day.
```

Additional test sentences include:

```text
La technologie change le monde.
Merci pour votre attention.
Le problème est très difficile.
```

### Portuguese → English

Example inputs include:

```text
este é o meu primeiro dia.
a tecnologia está mudando o mundo.
obrigado pela sua atenção.
o problema é muito difícil.
```

---

## 💻 Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Google Colab
* Transformer Architecture
* Neural Machine Translation

The notebook was configured to run with a **GPU**, and its Colab metadata specifies an **NVIDIA T4 GPU**.

---

## 📁 Project Structure

A recommended GitHub structure is:

```text
Translation-of-Sentence/
│
├── Translation_of_sentence.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   ├── ted_talks_en.csv
│   ├── ted_talks_pt-br.csv
│   ├── ted_talks_es.csv
│   └── ted_talks_fr.csv
│
└── images/
    └── model-results.png
```

If the datasets are too large for GitHub, keep them out of the repository and add instructions in the README explaining where to obtain them.

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Ramesh-sirvi/Translator-Model.git
cd Translation-of-Sentence
```

### 2. Install dependencies

```bash
pip install tensorflow pandas numpy matplotlib
```

### 3. Add the datasets

Place the required CSV files in the appropriate directory:

```text
data/
├── ted_talks_en.csv
├── ted_talks_pt-br.csv
├── ted_talks_es.csv
└── ted_talks_fr.csv
```

### 4. Open the notebook

You can run the project using:

* Google Colab
* Jupyter Notebook
* JupyterLab

For better training performance, a GPU is recommended.

---

## 🔮 Future Improvements

Possible improvements include:

* Train on a larger dataset.
* Increase the number of Transformer layers.
* Increase the model embedding dimension.
* Tune the learning rate and batch size.
* Implement a better tokenizer.
* Add validation and test datasets.
* Add BLEU or other translation evaluation metrics.
* Improve padding-mask handling.
* Use learning-rate scheduling.
* Implement beam search instead of simple greedy decoding.
* Train longer with appropriate regularization.
* Compare the custom Transformer with pretrained translation models.

---

## 📌 Limitations

The current implementation is primarily a learning/demo project. The generated translations can contain repeated or incorrect words, especially for Portuguese examples. The notebook's current predictions therefore should not be considered production-quality machine translation.

---

## 👨‍💻 Author

**Ramesh Choudhary**

This project demonstrates the implementation of a Transformer-based neural machine translation system using TensorFlow/Keras.

---

## ⭐ If You Like This Project

If this project helped you understand Transformer-based machine translation, consider giving the repository a ⭐ on GitHub.
