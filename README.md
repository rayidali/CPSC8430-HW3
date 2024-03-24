# CPSC8430-HW3
# Spoken-SQuAD Question Answering

This project focuses on developing a question answering system using the DistilBERT model and training it on the Spoken-SQuAD dataset. The goal is to build a model that can accurately answer questions based on the given spoken context.

## Dataset

The project utilizes the Spoken-SQuAD dataset, which is a spoken version of the SQuAD (Stanford Question Answering Dataset). The dataset consists of spoken passages and corresponding questions, along with their correct answers.

## Model

The question answering model is built using the DistilBERT architecture, which is a distilled version of the BERT (Bidirectional Encoder Representations from Transformers) model. DistilBERT is designed to be smaller and faster while retaining most of the performance of the original BERT model.

## Training

The model is trained using the PyTorch deep learning framework and the Transformers library by Hugging Face. The training process involves the following steps:
- Data loading and preprocessing
- Tokenization and encoding of the spoken passages and questions
- Fine-tuning the DistilBERT model on the Spoken-SQuAD dataset
- Evaluation of the trained model using the F1 score metric

## Results

After training for 3 epochs, the model achieved an F1 score of 61.63%.

## Usage

To use this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/spoken-squad-qa.git
2. Install Transformers and PyTorch:
   ```bash
   pip install torch transformers
3. Run the program:
   ```bash
   python hw3.py
