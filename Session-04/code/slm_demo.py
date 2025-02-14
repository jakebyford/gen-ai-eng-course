# pip install transformers
import socket
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

def generate_response(prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    return tokenizer.decode(output[0], skip_special_tokens=True)

prompts = [
    "What is the capital of France?",
    "Explain the benefits of using Small Language Models in applications",
]

HOST = "127.0.0.1"
PORT = 6000
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(5)

print(f"SLM Socket Service is running on {HOST} {PORT}...")

try:
    while True:
        client_socket, adddress = server_socket.accept()
        print(f"Connection from {adddress} has been established!")

        data = client_socket.recv(1024).decode("utf-8")
        print(f"Received data: {data}")

        try:
            json_data = json.loads(data).get("prompt", "")
            if not json_data:
                response = {"error": "Prompt can not be empty"}
            else:
                # LLM / SLM Inference integration
                response = generate_response(json_data)
                response = {"response": response}

        except Exception as e:
            response = {"error": str(e)}

        client_socket.sendall(json.dumps(response).encode("utf-8"))
        client_socket.close()

except Exception as e:
    print(f"An error occurred: {e}")

# for i, prompt in enumerate(prompts, 1):
#     print(f"Prompt {i}: {prompt}")
#     response = generate_response(prompt)
#     print(f"Response {i}: {response})}\n")