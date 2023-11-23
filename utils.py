# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:38:49 2023

@author: adywi
"""

import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
import transformers
from transformers import BertForSequenceClassification, AdamW, BertConfig,BertTokenizer,get_linear_schedule_with_warmup, AutoTokenizer
import torch
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import time
import datetime



class Data_Preprocessor:
    def __init__(self, data):
        self.data = data
        self.replace_space = re.compile('[/(){}\[\]\|@,;]')
        self.bad_symbols = re.compile('[^0-9a-z #+_]')
        self.stopwords = set(stopwords.words('english'))
                
    def preproc_data(self):
        data =  pd.read_csv(self.data)
        data["post"] = data["post"].apply(self.preprocess_text)
        return data
    
    def preprocess_text(self, data):
        text = BeautifulSoup(data, "lxml").text # HTML decoding
        text = text.lower() # lowercase text
        text = self.replace_space.sub(' ', text) # replace replace_space symbols by space in text
        text = self.bad_symbols.sub('', text) # delete symbols which are in bad_symbols from text
        text = ' '.join(word for word in text.split() if word not in self.stopwords) # delete stopwords from text
        return text
    
    def preproc_data_transformer(self):
        data =  pd.read_csv(self.data)
        data["post"] = data["post"].apply(lambda x: self.preproc_text_transformer(x))
        return data
     
    def preproc_text_transformer(self, text):
        sw = stopwords.words('english')
        text = text.lower()
        text = re.sub(r"[^a-zA-Z?.!,¿]+", " ", text) # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        text = re.sub(r"http\S+", "",text) #Removing URLs 
        #text = re.sub(r"http", "",text)
        html=re.compile(r'<.*?>')  
        text = html.sub(r'',text) #Removing html tag
        punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`" + '_'
        for p in punctuations:
            text = text.replace(p,'') #Removing punctuations         
        text = [word.lower() for word in text.split() if word.lower() not in sw]
        text = " ".join(text) #removing stopwords
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text) #Removing emojis
        return text
        
        

class Multi_Pipeline():
    def __init__(self, data, vectorizers, classifiers_param_grid, results_df = None, scoring='accuracy', cv=10):
       self.data = data
       self.vectorizers = vectorizers
       self.classifiers_param_grid = classifiers_param_grid
       self.scoring = scoring
       self.cv = cv
       if results_df is None:
           self.results_df = pd.DataFrame(columns=['Vectorizer', 'Classifier', 'Parameters', 'Accuracy'])
       else:
           self.results_df = results_df


    def fit_transform(self, X= None, y = None):
        if X is None or y is None:
            X, y = self.load_data()
        
        for vectorizer in self.vectorizers:
            for classifier, param_grid in self.classifiers_param_grid:
                pipeline = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', classifier)
                ])
                
                modified_param_grid = {
                'classifier__' + key: value for key, value in param_grid.items()
            }
                grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=modified_param_grid,
                scoring=self.scoring,
                cv=self.cv
                )

                grid_search.fit(X, y)

                result_dict = {
                    'Vectorizer': vectorizer,
                    'Classifier': classifier,
                    'Parameters': grid_search.best_params_,
                    'Accuracy': grid_search.best_score_
                }

                results_df = self.results_df.append(result_dict, ignore_index=True)
                
                print(f"Vectorizer: {vectorizer}, Classifier: {classifier}")
                print("Best Parameters:", grid_search.best_params_)
                print("Best Accuracy:", grid_search.best_score_)
                print("-------------------------------------------")
        
        results_df.to_excel('results.xlsx', index=False)

        return grid_search, results_df


    def load_data(self):
        X, y = self.data["post"], self.data["tags"]
        return X, y
    
    def fit_final_model(self):
        grid_search, results_df = self.fit_transform()
            
        X, y = self.load_data()
        
        best_params = results_df.iloc[results_df['Accuracy'].idxmax()]['Parameters']
        best_estimator = results_df.iloc[results_df['Accuracy'].idxmax()]['Classifier']
        best_vectorizer = results_df.iloc[results_df['Accuracy'].idxmax()]['Vectorizer']
        
        final_model = Pipeline([
            ('vectorizer', best_vectorizer),
            ('classifier', best_estimator)
        ])
        
        final_model.set_params(**best_params)
        final_model.fit(X, y)
        
        return final_model
    
    def fit_custom_model(self):
        if self.results_df is None:
            grid_search, results_df = self.fit_transform()
             
        X, y = self.load_data()
        
                        
class Transformer:
      def __init__(self, data, model, max_length, batch_size = 32):
          self.data = data
          self.posts = self.data["post"]
          self.labels = self.data["tags"]
          self.model = model
          self.batch_size = batch_size
          self.max_length = max_length
          self.tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path, do_lower_case=True)
          self.label_dict = {label: i for i, label in enumerate(set(self.labels))}
          self.reverse_label_dict = {i: label for label, i in self.label_dict.items()}
          self.numeric_labels = np.array([self.label_dict[label] for label in self.labels])

      
      def preproc(self):
          input_ids = []
          attention_masks = []
          
          for post in self.posts:
            encoded_dict = self.tokenizer.encode_plus(
                                post,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = self.max_length,           # Pad & truncate all sentences.
                                pad_to_max_length = True,
                                return_attention_mask = True,   # Construct attn. masks.
                                return_tensors = 'pt',     # Return pytorch tensors.
                                truncation = True # Truncat when longter than 512 tokens.
                           )
        
            # Add the encoded sentence to the list.    
            input_ids.append(encoded_dict['input_ids'])
        
            # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoded_dict['attention_mask'])
            
          input_ids = torch.cat(input_ids, dim=0)
          attention_masks = torch.cat(attention_masks, dim=0)
          labels = torch.tensor(self.numeric_labels)
          return input_ids, attention_masks, labels
       
      def data_loader(self):
        
        batch_size = self.batch_size
        val_ratio = 0.2

        input_ids, attention_masks, labels = self.preproc()
        # Indices of the train and validation splits stratified by labels
        train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size = val_ratio,
        shuffle = True,
        stratify = labels)
        
        # Train and validation sets
        train_set = TensorDataset(input_ids[train_idx], 
                          attention_masks[train_idx], 
                          labels[train_idx])
        
        val_set = TensorDataset(input_ids[val_idx], 
                        attention_masks[val_idx], 
                        labels[val_idx])


        # Create the DataLoaders for our training and validation sets.
        # We'll take training samples in random order. 
        train_dataloader = DataLoader(
                    train_set,  # The training samples.
                    sampler = RandomSampler(train_set), # Select batches randomly
                    batch_size = batch_size # Trains with this batch size.
                )
        
        # For validation the order doesn't matter, so we'll just read them sequentially.
        validation_dataloader = DataLoader(
                    val_set, # The validation samples.
                    sampler = SequentialSampler(val_set), # Pull out batches sequentially.
                    batch_size = batch_size # Evaluate with this batch size.
                )
        return train_dataloader, validation_dataloader
    
      def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)
    
      def format_time(self, elapsed):
        '''
        Takes a time in seconds and returns a string hh:mm:ss
        '''
        # Round to the nearest second.
        elapsed_rounded = int(round((elapsed)))
        # Format as hh:mm:ss
        return str(datetime.timedelta(seconds=elapsed_rounded))

      def train(self, epochs):  
        train_dataloader, validation_dataloader = self.data_loader()  
        training_stats = []

        # Measure the total training time for the whole run.
        total_t0 = time.time()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        optimizer = AdamW(self.model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
        # For each epoch...
        for epoch_i in range(0, epochs):
            
            # ========================================
            #               Training
            # ========================================
            # Perform one full pass over the training set.
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')
            # Measure how long the training epoch takes.
            t0 = time.time()
            total_train_loss = 0
            total_train_accuracy = 0
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, labels = batch
                optimizer.zero_grad()
                output = self.model(input_ids, 
                                     token_type_ids=None, 
                                     attention_mask=input_mask, 
                                     labels=labels)        
                loss = output.loss
                total_train_loss += loss.item()
                
                # Move logits and labels to CPU if we are using GPU
                logits = output.logits
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                
                total_train_accuracy += self.flat_accuracy(logits, label_ids)            
                # Perform a backward pass to calculate the gradients.
                loss.backward()
                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()
                # Update the learning rate.
                scheduler.step()
        
            # Calculate the average loss over all of the batches.
            avg_train_loss = total_train_loss / len(train_dataloader) 
            avg_train_accuracy = total_train_accuracy / len(validation_dataloader)
            
            # Measure how long this epoch took.
            training_time = self.format_time(time.time() - t0)
            print("")
            print("  Accuracy: {0:.2f}".format(avg_train_accuracy))           
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(training_time))
            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
            print("")
            print("Running Validation...")
            t0 = time.time()
            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            self.model.eval()
            # Tracking variables 
            total_eval_accuracy = 0
            best_eval_accuracy = 0
            total_eval_loss = 0
            nb_eval_steps = 0
            # Evaluate data for one epoch
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, labels = batch
                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():        
                    output= self.model(input_ids, 
                                           token_type_ids=None, 
                                           attention_mask=input_mask,
                                           labels=labels)
                loss = output.loss
                total_eval_loss += loss.item()
              
                # Move logits and labels to CPU if we are using GPU
                logits = output.logits
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()
                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches.
                total_train_accuracy += self.flat_accuracy(logits, label_ids)            
        
            # Report the final accuracy for this validation run.
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
            # Calculate the average loss over all of the batches.
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            # Measure how long the validation run took.
            validation_time = self.format_time(time.time() - t0)
            if avg_val_accuracy > best_eval_accuracy:
                torch.save(self.model, 'best_model')
                best_eval_accuracy = avg_val_accuracy
            #print("  Validation Loss: {0:.2f}".format(avg_val_loss))
            #print("  Validation took: {:}".format(validation_time))
            # Record all statistics from this epoch.
            training_stats.append(
                {
                    'epoch': epoch_i + 1,
                    'Training Loss': avg_train_loss,
                    'Valid. Loss': avg_val_loss,
                    'Valid. Accur.': avg_val_accuracy,
                    'Training Time': training_time,
                    'Validation Time': validation_time
                }
            )
        print("")
        print("Training complete!")
        
        print("Total training took {:} (h:mm:ss)".format(self.format_time(time.time()-total_t0)))      
      

def predict_test(final_model, test, file):
    y_pred = final_model.predict(test["post"])
    test.insert(2, "tags", y_pred)
    test.to_csv("Project/solution_1.csv", index = False)        
        


 
        
        
        
