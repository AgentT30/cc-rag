import os
import json
import boto3
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

def get_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(f"./assets/{pdf_path}")
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

while(True):
    choice = int(input("""Which credit card terms and conditions do you want to query today?
    1. Flipkart Axis Credit Card
    2. SBI Cashback Credit Card
    3. IDFC First Select Credit Card

    Please enter you choice: """))

    if choice not in [1, 2, 3]:
        print("Invalid choice entered, please enter the right choice!")
        print("------------------------------------------------------")

        continue
    break

credit_card_map = {
    1: "flipkart_axis_cc.pdf", 2: "sbi_cashback_cc.pdf", 3: "idfc_first_select_cc.pdf"
}

llm_model_map = {
    1: "gpt-4o-mini",
    2: "gpt-4o",
    3: "anthropic.claude-3-haiku-20240307-v1:0",
    4: "us.anthropic.claude-3-5-haiku-20241022-v1:0"
}

pdf_text = get_text_from_pdf(credit_card_map[choice])

system_prompt = """
You are a banker with an masters degree in finance and you are an expert at reading through terms and conditions of credit card and answering the questions asked by the user related to that credit card. You will answer in a precise and concise way. Remember the most important rule, the information that you give out to the user must strictly come from within the document and no other external factor. If you do find the requested information in the given document, then kindly answer with an 'The document does not outline the information that you requested' instead of creating an answer on your own. You should also cite the source text from which the information is retrieved.
"""

while(True):
    model_choice = int(input("""
    Which language model do you want to use today?
    1. OpenAI GPT 4o-mini
    2. OpenAI GPT 4o
    3. Anthropic Haiku 3
    4. Anthropic Haiku 3.5
                
    Please enter your choice: """))

    if not model_choice in [1,2,3,4]:
        continue

    break

query = input("You can now ask a question on the selected credit card using the selected model: ")

constructed_prompt = pdf_text + "\n\n ### \n" + f"The above is the terms and conditions document for a credit card. You will now read the above text and answer the question below along with the source citation text from the document: \n\nQuestion: {query}"

if 'gpt' in llm_model_map[model_choice]:
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    try:
        response = client.chat.completions.create(
            model=llm_model_map[model_choice],
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": constructed_prompt
                }
            ],
        )

        answer = response.choices[0].message.content
    except Exception as e:
        print("Oops something went wrong. Please try again")
elif 'claude' in llm_model_map[model_choice]:
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime', 
        region_name="us-east-1",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
    )

    content = [{
        "type": "text",
        "text": constructed_prompt
    }]

    
    message = {
        "role": "user",
        "content": content
    }

    body = json.dumps({
        "system": system_prompt,
        "messages": [message],
        "anthropic_version":  "bedrock-2023-05-31",
        "max_tokens": 4000,
        "temperature": 1,
        "top_p": 1,
        "top_k": 250,
    })

    try:
        response = bedrock_runtime.invoke_model(body=body, modelId=llm_model_map[model_choice])
        response_data = json.loads(response.get('body').read())
    except Exception as e:
        print("Oops something went wrong. Please try again")

    answer = response_data["content"][0]["text"]


print(f"Here is the answer for your query: \n\n{answer}")