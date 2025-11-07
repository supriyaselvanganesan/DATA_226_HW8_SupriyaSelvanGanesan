from airflow import DAG
from datetime import datetime, timedelta
from airflow.decorators import task
from airflow.models import Variable
import pandas as pd
import time
import requests
import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='Medium_to_Pinecone',
    default_args=default_args,
    description='Build a Medium Posting Search Engine using Pinecone',
    schedule_interval=timedelta(days=7),
    start_date=datetime(2025, 11, 5),
    catchup=False,
    tags=['medium', 'pinecone', 'search-engine'],
) as dag:

    @task
    def download_data():
        data_dir = '/tmp/medium_data'
        os.makedirs(data_dir, exist_ok=True)
        file_path = f"{data_dir}/medium_data.csv"
        url = 'https://s3-geospatial.s3.us-west-2.amazonaws.com/medium_data.csv'
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            with open(file_path, 'r') as f:
                line_count = sum(1 for _ in f)
            
            print(f"Downloaded file has {line_count} lines")
        else:
            raise Exception(f"Failed to download data: HTTP Status {response.status_code}")
        return file_path
    
    @task
    def preprocess_data(data_path):
        df = pd.read_csv(data_path)
        df['title'] = df['title'].astype(str).fillna('')
        df['subtitle'] = df['subtitle'].astype(str).fillna('')
        df['metadata'] = df.apply(lambda row: {'title': row['title'] + " " + row['subtitle']}, axis=1)
        df['id'] = df.reset_index(drop='index').index.astype(str)
        preprocessed_path = '/tmp/medium_data/medium_preprocessed.csv'
        df.to_csv(preprocessed_path, index=False)
        print(f"Preprocessed data saved to {preprocessed_path}")
        return preprocessed_path
    
    @task
    def create_pinecone_index():
        api_key = Variable.get("pinecone_api_key")
        pc = Pinecone(api_key=api_key)
        spec = ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
        index_name = 'semantic-search-fast'
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        if index_name in existing_indexes:
            pc.delete_index(index_name)
        pc.create_index(
            index_name,
            dimension=384,  
            metric='dotproduct',
            spec=spec
        )
        
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        
        print(f"Pinecone index '{index_name}' created successfully")
        return index_name
    
    @task
    def generate_embeddings_and_upsert(data_path, index_name):
        api_key = Variable.get("pinecone_api_key")
        df = pd.read_csv(data_path)
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("Generating embeddings...")
        batch_size = 100
        total_batches = (len(df) + batch_size - 1) // batch_size
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        for i in range(0, len(df), batch_size):
            print(f"Processing batch {i//batch_size + 1}/{total_batches}")
            batch_df = df.iloc[i:i+batch_size].copy()
            metadata_list = batch_df['metadata'].apply(eval).tolist()
            titles = [meta['title'] for meta in metadata_list]
            embeddings = model.encode(titles)
            upsert_data = []
            for j, (_, row) in enumerate(batch_df.iterrows()):
                upsert_data.append({
                    'id': str(row['id']),
                    'values': embeddings[j].tolist(),
                    'metadata': metadata_list[j]
                })
            
            index.upsert(upsert_data)
        print(f"Successfully upserted {len(df)} records to Pinecone")
        return index_name
    
    @task
    def test_search_query(index_name):
        api_key = Variable.get("pinecone_api_key")
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        pc = Pinecone(api_key=api_key)
        index = pc.Index(index_name)
        query = "what is ethics in AI"
        query_embedding = model.encode(query).tolist()
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        print(f"Search results for query: '{query}'")
        for result in results['matches']:
            print(f"ID: {result['id']}, Score: {result['score']}, Title: {result['metadata']['title'][:50]}...")
    
    # Define task dependencies using the TaskFlow API
    data_path = download_data()
    preprocessed_path = preprocess_data(data_path)
    print(preprocessed_path)
    index_name = create_pinecone_index()
    final_index_name = generate_embeddings_and_upsert(preprocessed_path, index_name)
    test_search_query(final_index_name)