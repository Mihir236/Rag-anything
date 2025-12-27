from openai import OpenAI

client = OpenAI(base_url="https://integrate.api.nvidia.com/v1")

try:
    response = client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": "Hello! Say: NVIDIA API is working perfectly."}],
        max_tokens=50
    )
    print("Success! Response:")
    print(response.choices[0].message.content)
except Exception as e:
    print("Error:")
    print(e)
