from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

system_prompt = """you are a doctor summary assistant.Generate Symptoms, Diagnosis, and Recommendation.Do not invent missing information."""

user_prompt = "Generate a doctor summary using the patient's details."

name = input("Enter patient name: ")
notes = input("Enter patient notes: ")

user_message = f"""
{user_prompt}

Patient Name: {name}
Patient Notes: {notes}
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
)

print("\nDoctor Summary:")
print("Patient Name:", name)
print(response.choices[0].message.content)