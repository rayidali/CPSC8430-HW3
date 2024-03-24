import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from tqdm import tqdm
import string
import re
import os
from collections import Counter

def load_squad_data(file_path):
    with open(file_path, 'rb') as f:
        squad_data = json.load(f)
    
    context_list, question_list, answer_list = [], [], []
    for data_group in squad_data['data']:
        for paragraph in data_group['paragraphs']:
            context = paragraph['context']
            for qa_pair in paragraph['qas']:
                question = qa_pair['question']
                if 'plausible_answers' in qa_pair:
                    answers = qa_pair['plausible_answers']
                else:
                    answers = qa_pair['answers']
                for answer in answers:
                    context_list.append(context)
                    question_list.append(question)
                    answer_list.append(answer)
    
    return context_list, question_list, answer_list

def preprocess_answers(answer_list, context_list):
    for answer, context in zip(answer_list, context_list):
        answer_text = answer['text']
        start_index = answer['answer_start']
        end_index = start_index + len(answer_text)
        
        if context[start_index:end_index] == answer_text:
            answer['answer_end'] = end_index
        else:
            for shift in [1, 2]:
                if context[start_index-shift:end_index-shift] == answer_text:
                    answer['answer_start'] = start_index - shift
                    answer['answer_end'] = end_index - shift

def tokenize_and_encode(tokenizer, context_list, question_list):
    encodings = tokenizer(context_list, question_list, truncation=True, padding=True)
    return encodings

def add_token_positions(encodings, answer_list, tokenizer):
    start_positions, end_positions = [], []
    for i in range(len(answer_list)):
        start_positions.append(encodings.char_to_token(i, answer_list[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answer_list[i]['answer_end']))
        
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answer_list[i]['answer_end'] - shift)
            shift += 1
    
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

class SquadDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    
    def __len__(self):
        return len(self.encodings.input_ids)

def train_model(model, dataloader, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch {epoch+1}')
            loop.set_postfix(loss=loss.item())

def save_model(model, tokenizer, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def evaluate_model(model, dataloader, tokenizer, device):
    model.eval()
    predicted_answers, true_answers = [], []
    loop = tqdm(dataloader)
    for batch in loop:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            
            for i in range(start_pred.shape[0]):
                all_tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][i])
                answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(all_tokens[start_pred[i] : end_pred[i]+1]))
                ref = tokenizer.decode(tokenizer.convert_tokens_to_ids(all_tokens[start_true[i] : end_true[i]+1]))
                predicted_answers.append(answer)
                true_answers.append(ref)
    
    return evaluate_predictions(true_answers, predicted_answers)

def normalize_answer(text):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

def compute_f1(prediction, truth):
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def evaluate_predictions(true_answers, predicted_answers):
    f1_scores = []
    for true_answer, predicted_answer in zip(true_answers, predicted_answers):
        f1_scores.append(compute_f1(predicted_answer, true_answer))
    
    return {'f1': 100.0 * sum(f1_scores) / len(f1_scores)}

def main():
    train_file = 'spoken_train-v1.1.json'
    val_file = 'spoken_test-v1.1.json'
    
    train_contexts, train_questions, train_answers = load_squad_data(train_file)
    val_contexts, val_questions, val_answers = load_squad_data(val_file)
    
    preprocess_answers(train_answers, train_contexts)
    preprocess_answers(val_answers, val_contexts)
    
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenize_and_encode(tokenizer, train_contexts, train_questions)
    val_encodings = tokenize_and_encode(tokenizer, val_contexts, val_questions)
    
    add_token_positions(train_encodings, train_answers, tokenizer)
    add_token_positions(val_encodings, val_answers, tokenizer)
    
    train_dataset = SquadDataset(train_encodings)
    val_dataset = SquadDataset(val_encodings)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased').to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
    
    train_model(model, train_loader, optimizer, device, num_epochs)
    
    output_dir = 'distilbert_squad_model'
    save_model(model, tokenizer, output_dir)
    
    eval_scores = evaluate_model(model, val_loader, tokenizer, device)
    print(f"Evaluation scores: {eval_scores}")

if __name__ == '__main__':
    main()