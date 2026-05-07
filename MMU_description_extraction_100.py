import pandas as pd
from google import genai
import os

config = {'gcp_gemini_parameters': {'project': 'prj-orica-ai-dev', 'model_name': 'gemini-2.5-pro', 'location': 'us-central1'}}

def gemini_text_action(config, prompt):
    client = genai.Client(vertexai=True, project=config['gcp_gemini_parameters']['project'], location=config['gcp_gemini_parameters']['location'])
    return client.models.generate_content(model=config['gcp_gemini_parameters']['model_name'], contents=[prompt]).text

def extract_batch():
    df = pd.read_csv('data/MMU_next_100_samples.csv')
    all_terms = set()
    prompt_template = "Identify and list all phrases/nouns in the caption referring to the Mobile Manufacturing Unit (MMU). Output ONLY terms separated by semicolons. Caption: {caption}"
    
    for _, row in df.iterrows():
        print(f"Extracting: {row['image_file_path']}")
        try:
            res = gemini_text_action(config, prompt_template.format(caption=row['caption']))
            all_terms.update([t.strip().lower() for t in res.split(';') if t.strip()])
        except Exception as e: print(f"Error: {e}")
    
    os.makedirs('outputs', exist_ok=True)
    pd.DataFrame(list(all_terms), columns=['mmu_generic_descriptions']).to_csv('outputs/MMU_description_100.csv', index=False)
    print("Extraction complete. Saved to outputs/MMU_description_100.csv")

if __name__ == "__main__":
    extract_batch()