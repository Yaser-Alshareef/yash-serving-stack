from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

resp = client.chat.completions.create(
    model="Qwen2.5-1.5B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "You are a terse assistant."
        },
        {
            "role": "user",
            "content": "Name three primary colours."
        }
    ],
    max_tokens=64
)

print("Reply:", resp.choices[0].message.content)
print("Finish reason:", resp.choices[0].finish_reason)
print("Usage:", resp.usage)