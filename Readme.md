# Property Deed Information Extractor

A RunPod serverless endpoint for extracting information from property deed PDFs using OCR and GPT-4.

## Features
- PDF text extraction using DocTR OCR
- Information extraction using GPT-4
- GPU-accelerated processing
- Serverless deployment on RunPod

## Input Format
```json
{
    "input": {
        "base64_pdf": "base64_encoded_pdf_string"
    }
}
```

## Output Format
```json
{
    "success": true,
    "data": {
        "owner_name": "...",
        "property_address": "...",
        "property_parcel_id": "..."
        // ... other fields
    },
    "error": null
}
```

## Deployment
1. Fork this repository
2. Set up RunPod account
3. Add OPENAI_API_KEY to RunPod secrets
4. Deploy as serverless endpoint

## Requirements
- CUDA-capable GPU
- OpenAI API key
- RunPod account