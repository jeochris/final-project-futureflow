from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

import json
from tqdm import tqdm

from generate_blip2 import generate_blip2
from lexical_constraints import init_batch

class Runner:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map="auto"
        )
        self.device = device
        self.processor = processor
        self.model = model

    def __call__(self, image_path, constraints_list=None):
        image = Image.open(image_path)

        inputs = self.processor(images=image, return_tensors="pt").to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs, num_beams=20)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # print('original generate function without constraint:', generated_text)
        res1 = generated_text

        #########################################################

        # special token id 모음
        period_id = [self.processor.tokenizer.convert_tokens_to_ids('.')]
        period_id.append(self.processor.tokenizer.convert_tokens_to_ids('Ġ.'))
        eos_ids = [self.processor.tokenizer.eos_token_id] + period_id
        #print(eos_ids)
        PAD_ID = self.processor.tokenizer.convert_tokens_to_ids('<pad>')

        def tokenize_constraints_chk(tokenizer, raw_cts):
            def tokenize2(phrase):
                tokens = tokenizer.tokenize(phrase)
                #print(phrase)
                token_ids = tokenizer.convert_tokens_to_ids(tokens)
                return token_ids, True
            return [[list(map(tokenize2, clause)) for clause in ct] for ct in raw_cts]
        
        # constraints_list = [[[" glasses"]]] # 앞에 띄어쓰기!
        #constraints_list = [[[" game", " games"], [" league"], [" exciting", " exicted"]]]
        constraints_list = tokenize_constraints_chk(self.processor.tokenizer, constraints_list)
        #print(constraints_list)

        constraints = init_batch(raw_constraints=constraints_list,
                                    beam_size=20,
                                    eos_id=eos_ids)

        new_generated_ids = generate_blip2(self.model, **inputs, new_constraints=constraints)
        new_generated_text = self.processor.batch_decode(new_generated_ids, skip_special_tokens=True)[0].strip()
        # print('new generate function with constraint:', new_generated_text)

        res2 = new_generated_text
        return res1, res2

if __name__ == "__main__":
    fx = Runner()

    result = {}
    with open('./data/cic.json') as f:
        cic = json.load(f)
    
    for fileName in tqdm(list(cic.keys())):
        result[fileName] = {}
        constraints_list = [[[" "+const] for const in cic[fileName]['constraint']]]
        res1, res2 = fx(image_path='./data/images/'+fileName, constraints_list=constraints_list)

        result[fileName]['without constraint'] = res1
        result[fileName]['with constraint'] = res2

        with open('./result.json', 'w') as outfile:
            json.dump(result, outfile)

    with open('./result.json', 'w') as outfile:
        json.dump(result, outfile)