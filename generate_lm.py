import torch
from typing import Optional

import copy
from transformers import GenerationConfig, LogitsProcessorList

from constrained_beam_search import _generate_beam_search

@torch.no_grad()
def generate_lm(
    self,
    inputs: Optional[torch.Tensor] = None,
    generation_config: Optional[GenerationConfig] = None,
    **kwargs
):
    # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    self._validate_model_class()
    # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    if generation_config is None:
        generation_config = self.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    generation_config.validate()

    #self._validate_model_kwargs(model_kwargs.copy())
    # 2. Set generation parameters if not already defined
    logits_processor = LogitsProcessorList()
    
    # 3. Define model inputs
    # inputs_tensor has to be defined
    # model_input_name is defined if model-specific keyword input is passed
    # otherwise model_input_name is None
    # all model-specific keyword inputs are removed from `model_kwargs`
    inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
        inputs, generation_config.bos_token_id, model_kwargs
    )
    # print(inputs_tensor)
    # print(model_input_name)
    # print(model_kwargs)
    batch_size = inputs_tensor.shape[0]
    # 4. Define other model kwargs
    model_kwargs["output_attentions"] = generation_config.output_attentions
    model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    model_kwargs["use_cache"] = generation_config.use_cache
    input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")
    
    input_ids_seq_length = input_ids.shape[-1]
    # 8. prepare distribution pre_processing samplers
    logits_processor = self._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=inputs_tensor,
        prefix_allowed_tokens_fn=None,
        logits_processor=logits_processor,
    )

    input_ids, model_kwargs = self._expand_inputs_for_generation(
        input_ids=input_ids,
        expand_size=generation_config.num_beams,
        is_encoder_decoder=self.config.is_encoder_decoder,
        **model_kwargs,
    )
    # print(input_ids)
    # print(model_kwargs)
    
    # # create attention mask if necessary
    # if hasattr(self.config, "vocab_size"):
    #     vocab_size = self.config.vocab_size
    # # set effective batch size and effective batch multiplier according to do_sample
    # if generation_config.do_sample:
    #     effective_batch_size = batch_size * generation_config.num_return_sequences
    #     effective_batch_mult = generation_config.num_return_sequences
    # else:
    #     effective_batch_size = batch_size
    #     effective_batch_mult = 1

    # attention_mask = model_kwargs.pop("attention_mask")
    # # Expand input ids if num_beams > 1 or num_return_sequences > 1
    # if generation_config.num_return_sequences > 1 or generation_config.num_beams > 1:
    #     input_ids_len = input_ids.shape[-1]
    #     input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * generation_config.num_beams, input_ids_len)
    #     attention_mask = attention_mask.unsqueeze(1).expand(
    #         batch_size, effective_batch_mult * generation_config.num_beams, input_ids_len
    #     )
    #     input_ids = input_ids.contiguous().view(
    #         effective_batch_size * generation_config.num_beams, input_ids_len
    #     )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
    #     attention_mask = attention_mask.contiguous().view(
    #         effective_batch_size * generation_config.num_beams, input_ids_len
    #     )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
    encoder_outputs = None
    cur_len = input_ids.shape[-1]

    # attention_mask -> model_kwargs 안에 그대로
    # encoder_outputs, cur_len, vocab_size 따로 지정 X

    output = _generate_beam_search(
        self,
        input_ids=input_ids,
        logits_processor=logits_processor,
        cur_len=cur_len,
        max_length=generation_config.max_length,
        min_length=generation_config.min_length,
        do_sample=generation_config.do_sample,
        early_stopping=generation_config.early_stopping,
        temperature=generation_config.temperature,
        top_k=generation_config.top_k,
        top_p=generation_config.top_p,
        repetition_penalty=generation_config.repetition_penalty,
        no_repeat_ngram_size=generation_config.no_repeat_ngram_size,
        bad_words_ids=generation_config.bad_words_ids,
        bos_token_id=generation_config.bos_token_id,
        pad_token_id=generation_config.pad_token_id,
        decoder_start_token_id=None,
        eos_token_id=generation_config.eos_token_id,
        batch_size=1,
        num_return_sequences=generation_config.num_return_sequences,
        length_penalty=generation_config.length_penalty,
        num_beams=generation_config.num_beams,
        vocab_size=self.config.vocab_size,
        encoder_outputs=encoder_outputs,
        attention_mask=model_kwargs['attention_mask'],
        use_cache=model_kwargs['use_cache'],
        constraints=model_kwargs['new_constraints'],
        prune_factor=model_kwargs['prune_factor'],
        sat_tolerance=model_kwargs['sat_tolerance'],
        beta=model_kwargs['beta'],
        early_stop=model_kwargs['early_stop'],
        model_specific_kwargs=model_kwargs,
    )

    return output