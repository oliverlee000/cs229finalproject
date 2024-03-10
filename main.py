import pandas as pd
import openai
# Passkey : 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
def main():
    # Load the CSV data
    data = pd.read_csv('your_data.csv')
    # Prompt Chat GPT
    open.api_key = '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'
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100)
    print(response.choices[0].text.strip())
