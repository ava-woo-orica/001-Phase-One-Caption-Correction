import pandas as pd
import os
from google import genai

# --- 1. CONFIGURATION ---
config = {
    'gcp_gemini_parameters': {
        'project': 'prj-orica-ai-dev',
        'model_name': 'gemini-2.5-pro',
        'location': 'us-central1'
    }
}

def gemini_text_action(config, prompt):
    client = genai.Client(vertexai=True, project=config['gcp_gemini_parameters']['project'], location=config['gcp_gemini_parameters']['location'])
    response = client.models.generate_content(model=config['gcp_gemini_parameters']['model_name'], contents=[prompt])
    return response.text

def refine_captions_mmu(csv_path, mmu_descriptions_path):
    df = pd.read_csv(csv_path)
    mmu_terms_df = pd.read_csv(mmu_descriptions_path)
    mmu_terms_str = ", ".join(mmu_terms_df['mmu_generic_descriptions'].tolist())

    refinement_prompt_template = """

    You are a technical editor. Your goal is to correctly label vehicles: either as a "mobile manufacturing unit (MMU)" or a "white service truck" while maintaining natural English flow.

    GENERIC TERMS TO REPLACE: [{mmu_list}]

    RULES FOR IDENTIFICATION:
    1. SPATIAL/PRIMARY RULE: If a white truck or large vehicle is described as "dominating the foreground," "on the right side," or is the "primary subject," it is DEFINITELY a mobile manufacturing unit (MMU).
    2. SECONDARY VEHICLES: If there are other white trucks/vehicles in the background or periphery, identify if they are standard utility/pickup trucks. If they are, label them as "white service truck" instead of MMU.
    
    RULES FOR REPLACEMENT FLOW:
    1. Identify the generic terms from the list above in the Caption.
    2. For the MMU: Replace the FIRST mention with "mobile manufacturing unit (MMU)". For all subsequent mentions of THAT specific vehicle, use "the MMU", "the unit", or "it" to avoid repetitive, clunky phrasing.
    3. For Service Trucks: If a vehicle is identified as secondary/utility per Rule 2, replace it with the phrase "white service truck" and DO NOT use the term MMU for it.
    4. CRITICAL: Avoid redundancy. If one sentence mentions the same vehicle twice (e.g., "The truck and the vehicle's auger"), do NOT use the full phrase twice. Rewrite as "The mobile manufacturing unit (MMU) and its auger."
    5. Do NOT change the technical tone or style of the caption or any other part of the sentence.
    
    Caption: {caption}
    """
    

    refined_captions = []
    for index, row in df.iterrows():
        print(f"Refining: {row['basename']}")
        prompt = refinement_prompt_template.format(mmu_list=mmu_terms_str, caption=row['captions'])
        try:
            refined_text = gemini_text_action(config, prompt)
            refined_captions.append(refined_text.strip())
        except Exception as e:
            print(f"Error on {row['basename']}: {e}")
            refined_captions.append(row['captions']) 

    df['refined_caption'] = refined_captions
    
    os.makedirs('outputs', exist_ok=True)
    output_file = 'outputs/validated_10_training_samples_refined_mmu.csv'
    df.to_csv(output_file, index=False)
    print(f"\nFinal refined CSV saved at: {output_file}")

if __name__ == "__main__":
    refine_captions_mmu('data/validated_10_training_samples_final.csv', 'outputs/MMU_description.csv')