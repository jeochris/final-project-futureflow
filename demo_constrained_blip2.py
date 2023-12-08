import gradio as gr

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from generate_blip2 import generate_blip2
from lexical_constraints import init_batch


device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto"
)


def apply_model(img, user_constraints):
    inputs = processor(images=img, return_tensors="pt").to(device, torch.float16)

    # 공통 config 설정
    inputs['num_beams'] = 20
    inputs['no_repeat_ngram_size'] = 3
    inputs['length_penalty'] = 0.5

    # Blip2
    generated_ids_1 = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids_1, skip_special_tokens=True)[0].strip()

    # special token id 모음
    period_id = [processor.tokenizer.convert_tokens_to_ids('.')]
    period_id.append(processor.tokenizer.convert_tokens_to_ids('Ġ.'))
    eos_ids = [processor.tokenizer.eos_token_id] + period_id

    PAD_ID = processor.tokenizer.convert_tokens_to_ids('<pad>')

    def tokenize_constraints_chk(tokenizer, raw_cts):
        def tokenize2(phrase):
            tokens = tokenizer.tokenize(phrase)
            token_ids = tokenizer.convert_tokens_to_ids(tokens)
            return token_ids, True
        return [[list(map(tokenize2, clause)) for clause in ct] for ct in raw_cts]


    user_constraint = user_constraints.split(', ')
    combine_list = [[[" " + x] for x in user_constraint]]

    constraints_list = tokenize_constraints_chk(processor.tokenizer, combine_list)

    constraints = init_batch(raw_constraints=constraints_list,
                                beam_size=20,
                                eos_id=eos_ids)

    new_generated_ids = generate_blip2(model, **inputs, new_constraints=constraints)
    new_generated_text = processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0].strip()

    return generated_text, new_generated_text


css = """
.out textarea {font-size: 25px}
.out span {font-size: 25px}
"""

demo = gr.Blocks(css=css)

with demo:
    with gr.Row():
        img_input = gr.Image(type="pil", label="input image")
        text_input = gr.Textbox(label="Constraint")

    b3 = gr.Button("Generate Captions")

    blip2_output = gr.Textbox(label="Vanilla Blip2")
    blip2_con_output = gr.Textbox(label="Constrained Blip2")

    b3.click(apply_model, inputs=[img_input, text_input], outputs=[blip2_output, blip2_con_output])

demo.launch(debug=True)