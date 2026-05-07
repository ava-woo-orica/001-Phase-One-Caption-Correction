import pandas as pd
from google import genai
import os

# --- 1. CONFIGURATION ---
config = {
    'gcp_gemini_parameters': {
        'project': 'prj-orica-ai-dev',
        'model_name': 'gemini-2.5-pro', 
        'location': 'us-central1'
    }
}

# --- 2. TEXT-ONLY GEMINI FUNCTION ---
def gemini_text_action(config, prompt):
    project = config['gcp_gemini_parameters']['project']
    model_name = config['gcp_gemini_parameters']['model_name']
    location = config['gcp_gemini_parameters']['location']
    
    client = genai.Client(vertexai=True, project=project, location=location)

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
    )
    return response.text

# --- 3. EXTRACTION PHASE ---
def extract_mmu_terms_text_only(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find the CSV at: {os.path.abspath(csv_path)}")

    df = pd.read_csv(csv_path)
    all_extracted_terms = set()

    extraction_prompt_template = """
    Identify and list all phrases, nouns, or descriptions in the provided caption 
    that refer to the Mobile Manufacturing Unit (MMU).
    
    Output ONLY the list of terms separated by semicolons.
    
    Caption: {caption}
    """

    for index, row in df.iterrows():
        print(f"Processing: {row['basename']}")
        prompt = extraction_prompt_template.format(caption=row['captions'])
        
        try:
            raw_output = gemini_text_action(config, prompt)
            terms = [t.strip().lower() for t in raw_output.split(';') if t.strip()]
            all_extracted_terms.update(terms)
        except Exception as e:
            print(f"Error on {row['basename']}: {e}")

    mmu_df = pd.DataFrame(list(all_extracted_terms), columns=['mmu_generic_descriptions'])
    
    # Save to outputs folder
    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/MMU_description.csv'
    mmu_df.to_csv(output_path, index=False)
    print(f"\nExtraction Complete. File saved as '{os.path.abspath(output_path)}'")
    return mmu_df

if __name__ == "__main__":
    target_csv = 'data/validated_10_training_samples_final.csv'
    extract_mmu_terms_text_only(target_csv)