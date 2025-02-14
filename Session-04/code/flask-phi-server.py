# pip install transformers flask

from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

model_name = "microsoft/phi-1_5"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        json_data = request.json.get("prompt", "")
        print(f"Received data: {json_data}")
        if not json_data:
            response = {"error": "Prompt can not be empty"}
        else:
            input_ids = tokenizer.encode(json_data, return_tensors="pt")
            output = model.generate(input_ids, max_length=50, num_return_sequences=1)
            response = {"response": tokenizer.decode(output[0], skip_special_tokens=True)}
    except Exception as e:
        response = {"error": str(e)}
    return jsonify({"response": response})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)