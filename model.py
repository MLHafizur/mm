from os.path import exists
import os
import time
import sys
from os import path
import joblib
import json
import io
import pandas as pd
from typing import Dict, List
from mlserver import MLModel
from mlserver.utils import get_model_uri
from mlserver.codecs.pandas import PandasCodec
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput, Parameters
from multi_classification import predict, prepare_dataset, classify_data
#from classification import process_adverse_types_categories, split_more_text_data
from args import Args
import torch
from azure.storage.blob import BlobServiceClient
print("imported")
WELLKNOWN_MODEL_FILENAMES = ["model.pt"]

STORAGEACCOUNTURL = "https://devussccrsv3.blob.core.windows.net"
STORAGEACCOUNTKEY = "Spq7/2IG0y1P5j6pvAsOPFGEJjFclKDiEzghmf7Vkp9gfRl67xnqZJNYZT8onM2ZvOc2DmVwADkjgqpYej2Y+w=="
CONTAINERNAME = "models"
blob_service_client_instance = BlobServiceClient(account_url=STORAGEACCOUNTURL, credential=STORAGEACCOUNTKEY)
container_client = blob_service_client_instance.get_container_client(container=CONTAINERNAME)


class MulticlassAAll(MLModel):
    async def load(self) -> bool:
        model_load_start = time.time()
        model_uri = await get_model_uri(self._settings)
        print("model_uri", model_uri)
        model_path = path.join(model_uri, WELLKNOWN_MODEL_FILENAMES[0])
        print("model_path", model_path)
        self.model = torch.load(model_uri, map_location=torch.device('cpu'))
        print("model", self.model)
        self.model.eval()
        model_load_finish = time.time()
        print("Model loaded in {} seconds".format(model_load_finish - model_load_start))
        return True
    
    async def predict(self, payloads: InferenceRequest) -> InferenceResponse:

        args = Args()
        args = args.update_args()
        args.mode = 'predict'    
        args.load_model = True
      
        # loading label encoder
        label_encoder_adverse_all = joblib.load("./src/data/registered-models/label_encoder_adverse_all.pkl")
        print("label_encoder_adverse_all loaded")
       

        print("Reading data")
        data_loading_start = time.time()
        df = self._extract_inputs(payloads)
        #df = PandasCodec.decode_request(payloads)
        print(df.head())
        data_loading_finish = time.time()
        print("Data loaded in {} seconds".format(data_loading_finish - data_loading_start))

        
        print("input data pre-processing")
        data_preprocessing_start = time.time()
        df_excluded = df[df['excluded'] != 0]
        df = df[df['excluded'] == 0]
        print(f"Excluded records: {df_excluded.shape}")
        print(f"Non-excluded records: {df.shape}")
        df = df[df['full_text'].apply(lambda x: isinstance(x, str))].reset_index(drop=True)
        sentences = list(df['full_text'].values)
        _, adv_dataloader = prepare_dataset(args, self.model, sentences, shuffle=False)
        data_preprocessing_finish = time.time()
        print("Data pre-processed in {} seconds".format(data_preprocessing_finish - data_preprocessing_start))

        print("Predicting")
        prediction_start = time.time()
        adv_predictions = pd.DataFrame(predict(args, self.model, adv_dataloader, label_encoder_adverse_all))
        prediction_finish = time.time()
        print("Prediction finished in {} seconds".format(prediction_finish - prediction_start))

        print("append scores to original dataframe")
        records = pd.concat([df, adv_predictions], axis=1)


        print("create post-prediction flags")
        post_processing_start = time.time()
        labels = label_encoder_adverse_all.inverse_transform(list(map(int,list(records['top1_label'].values))))
        records['top1_label']=labels
        print("Checkpoint-2")
        labels = label_encoder_adverse_all.inverse_transform(list(map(int,list(records['top2_label'].values))))
        records['top2_label']=labels
        labels = label_encoder_adverse_all.inverse_transform(list(map(int,list(records['top3_label'].values))))
        records['top3_label']=labels
        labels = label_encoder_adverse_all.inverse_transform(list(map(int,list(records['top4_label'].values))))
        records['top4_label']=labels
        print("Checkpoint-3")
        records['adv_nonadv_label']=0
        records['adv_nonadv_label'][records['adverse']==1]="Adverse"
        records['adv_nonadv_label'][records['adverse']==0]="Nonadverse"
        records['adv_nonadv_label_confidence']=records['adverse_probas'].apply(lambda x:[x])
        records['adverse_types_predicted_label_top2'] = records['top1_label'].apply(lambda x:[x]) + records['top2_label'].apply(lambda x:[x])+ records['top3_label'].apply(lambda x:[x])+ records['top4_label'].apply(lambda x:[x])
        records['adverse_types_predicted_label_top2_confidence'] = records['top1_probas'].apply(lambda x:[x]) + records['top2_probas'].apply(lambda x:[x])+ records['top3_probas'].apply(lambda x:[x]) + records['top4_probas'].apply(lambda x:[x])
        print("Checkpoint-4")
        records['adverse_types_predicted_label_top2'],records['adverse_types_predicted_label_top2_confidence'] = self.process_adverse_types_categories(records['adverse_types_predicted_label_top2'].values,records['adverse_types_predicted_label_top2_confidence'].values)
        print("merge complete result df")
        results = pd.concat([records, df_excluded])
        post_processing_end = time.time()
        print("Post-processing finished in {} seconds".format(post_processing_end - post_processing_start))

        print("Generating json output")
        output = results.to_json(orient="records", lines=True)
        
        return InferenceResponse(
            model_name = self.name,
            model_version=self.version,
            outputs=[
                ResponseOutput(
                    name="Predictions",
                    shape=[len(output)],
                    datatype="BYTES",
                    data=output
                )
            ], 
        )
    
    def _extract_inputs(self, payloads: InferenceRequest) -> pd.DataFrame:

        print("Initiating data extraction")

        input = pd.DataFrame()

        for inp in payloads.inputs:
            print("Framing")
            if inp.name == "TF_Idf_score":
                input[inp.name] = inp.data
            if inp.name == "bm25":
                input[inp.name] = inp.data
            if inp.name == "deduplication":
                input[inp.name] = inp.data
            if inp.name == "description":
                input[inp.name] = inp.data
            if inp.name == "duplicate_content":
                input[inp.name] = inp.data
            if inp.name == "duplicates":
                input[inp.name] = inp.data
            if inp.name == "duplicates_message":
                input[inp.name] = inp.data
            if inp.name == "excluded":
                input[inp.name] = inp.data
            if inp.name == "excluded_keyword_content":
                input[inp.name] = inp.data
            if inp.name == "flag_original_record":
                input[inp.name] = inp.data
            if inp.name == "full_name_match":
                input[inp.name] = inp.data
            if inp.name == "full_text":        
                input[inp.name] = inp.data
            if inp.name == "id":
                input[inp.name] = inp.data
            if inp.name == "jurisdiction":
                input[inp.name] = inp.data
            if inp.name == "language":
                input[inp.name] = inp.data
            if inp.name == "language_id":
                input[inp.name] = inp.data
            if inp.name == "language_score":
                input[inp.name] = inp.data
            if inp.name == "languages_fasttex":
                input[inp.name] = inp.data
            if inp.name == "model_preds":        
                input[inp.name] = inp.data
            if inp.name == "partial_name_matc":        
                input[inp.name] = inp.data
            if inp.name == "record_number":        
                input[inp.name] = inp.data
            if inp.name == "score":        
                input[inp.name] = inp.data
            if inp.name == "scores_1":        
                input[inp.name] = inp.data
            if inp.name == "source_name":        
                input[inp.name] = inp.data
            if inp.name == "source_type":        
                input[inp.name] = inp.data
            if inp.name == "text":        
                input[inp.name] = inp.data
            if inp.name == "tfidf_cossim":        
                input[inp.name] = inp.data
            if inp.name == "title":        
                input[inp.name] = inp.data
            if inp.name == "unique_query_fields_count":        
                input[inp.name] = inp.data
            if inp.name == "url":        
                input[inp.name] = inp.data
        print("done")
        print(input.shape)
        return input

    def process_adverse_types_categories(self, predictions, probabilities):
        '''As Research Clarity, when an adverse results is displayed in the Adverse tab, then the top 4 class predictions for the results can be selected to be displayed 
        so that when the top 2 predictions are displayed if the second one is Nonadverse or Ignore, they can be skipped and instead the 4th adverse class can be shown. 
        This way we can avoid displaying Nonadverse and Ignore classes in the Adverse Tab and given that these classes do not provide any context to the user and can be confusing. 
        User story: 171598 (Q1S3 2022)
        a,b- list of lists
        returns two list of lists'''
        preds=[]
        probs=[]
        for i,value in enumerate(predictions):
            if predictions[i]!=None:
                a = predictions[i]
                b = probabilities[i]
                
                if value[1] in ['Non Adverse','Ignore'] and value[2] in ['Non Adverse','Ignore']:
                    del a[1]
                    del b[1]
                    del a[1]
                    del b[1]
                    
                elif value[1] == 'Non Adverse' and value[0]!='Ignore':
                    #print(i)
                    #print(value)
                    del a[1]
                    del b[1]
                elif value[1] == 'Ignore' and value[0]!='Non Adverse':
                    #print(value)
                    del a[1]
                    del b[1]
                
                else:
                    del a[2]
                    del b[2]

                preds.append(a[:2])
                probs.append(b[:2])
            else:
                preds.append(None)
                probs.append(None)
        return preds, probs