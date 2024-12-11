import runpod
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json
from openai import OpenAI
import os
import io
import re
import base64

# Initialize the OCR model with CUDA support
try:
    model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True).cuda()
except Exception as e:
    print(f"Error initializing OCR model: {str(e)}")
    raise

# Initialize OpenAI client with better error handling
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment variables")
except Exception as e:
    print(f"Error initializing OpenAI client: {str(e)}")
    raise

def extract_text_from_pdf(pdf_file_bytes):
    try:
        doc = DocumentFile.from_pdf(pdf_file_bytes)
        result = model(doc)
        return result
    except Exception as e:
        print(f"Error in PDF extraction: {str(e)}")
        raise

def clean_extracted_text(result):
    extracted_text = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                line_text = ' '.join(word.value for word in line.words)
                extracted_text.append(line_text)
    return '\n'.join(extracted_text)

def extract_critical_information(text):
    try:
        MODEL = "gpt-4o"  # Using the original model
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": """You are a property deed expert attorney for the US with the following responsibilities:
                    1. Extract only explicitly stated information from property deeds
                    2. Mark missing information as "Not specified in document"
                    3. Maintain exact legal language for crucial elements
                    4. Flag any ambiguities or inconsistencies
                    5. Never infer or assume information not present in the document"""},
                {"role": "user", "content": f"Given the raw information extracted using OCR from a PDF, you should extract the most important parts of the deed such as the owner's name, property parcel id, address, and any other important factors. The OCR results are stored in the variable `result`.\n\nOCR Result:\n{text}\n\nPlease provide the extracted information in JSON format.\n\nExample JSON output:\n{{\n  \"owner_name\": \"\",\n  \"property_address\": \"\",\n  \"property_parcel_id\": \"\",\n  \"document_id\": \"\",\n  \"legal_description\": \"\",\n  \"grantor_name\": \"\",\n  \"grantee_name\": \"\",\n  \"deed_type\": \"\",\n  \"liens_and_encumbrances\": \"\",\n  \"signatures\": \"\",\n  \"notarization_details\": \"\",\n  \"recording_information\": \"\",\n  \"consideration\": \"\",\n  \"habendum_clause\": \"\",\n  \"warranty_clauses\": \"\",\n  \"tax_information\": \"\",\n  \"title_insurance_details\": \"\"\n}}Extracted Information JSON:, Warning: Do not make up fake information"}],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenAI extraction: {str(e)}")
        raise

def clean_and_convert_to_json(input_string):
    # Remove markdown code block indicators and whitespace
    cleaned_string = input_string.strip()
    cleaned_string = re.sub(r'^```json\s*|\s*```$', '', cleaned_string, flags=re.MULTILINE)
    
    # Remove non-printable characters except newlines
    cleaned_string = ''.join(char for char in cleaned_string if char.isprintable() or char in '\n\r')
    
    # Ensure proper JSON structure
    cleaned_string = cleaned_string.strip()
    if not cleaned_string.startswith('{'):
        cleaned_string = '{' + cleaned_string
    if not cleaned_string.endswith('}'):
        cleaned_string = cleaned_string + '}'
    
    try:
        data = json.loads(cleaned_string)
        return data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Problematic string:\n{cleaned_string}")
        return {"error": f"JSON parsing error: {str(e)}"}

def handler(event):
    """
    RunPod handler function
    """
    try:
        # Get input data
        job_input = event["input"]
        
        if "base64_pdf" not in job_input:
            return {"error": "No PDF data provided"}
        
        # Decode base64 PDF
        try:
            pdf_bytes = base64.b64decode(job_input["base64_pdf"])
        except Exception as e:
            return {"error": f"Invalid base64 PDF data: {str(e)}"}
        
        # Process the PDF
        result = extract_text_from_pdf(io.BytesIO(pdf_bytes))
        cleaned_text = clean_extracted_text(result)
        extracted_info = extract_critical_information(cleaned_text)
        json_result = clean_and_convert_to_json(extracted_info)
        
        return {
            "success": True,
            "data": json_result,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "data": None,
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})