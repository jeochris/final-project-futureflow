import torch
from typing import Optional

from generate_lm import generate_lm

@torch.no_grad()
def generate_blip2(
    self,
    pixel_values: torch.FloatTensor,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    **generate_kwargs,
) -> torch.LongTensor:
    """
    Overrides `generate` function to be able to use the model as a conditional generator.
    Args:
        pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
            Input images to be processed.
        input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
            The sequence used as a prompt for the generation.
        attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
            Mask to avoid performing attention on padding token indices
    Returns:
        captions (list): A list of strings of length batch_size * num_captions.
    """

    if hasattr(self, "hf_device_map"):
        # preprocess for `accelerate`
        self._preprocess_accelerate()

    batch_size = pixel_values.shape[0]
    image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state
    image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_outputs = self.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_attention_mask,
        return_dict=True,
    )
    query_output = query_outputs.last_hidden_state

    language_model_inputs = self.language_projection(query_output)
    language_attention_mask = torch.ones(
        language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
    )
    if input_ids is None:
        input_ids = (
            torch.LongTensor([[self.config.text_config.bos_token_id]])
            .repeat(batch_size, 1)
            .to(image_embeds.device)
        )
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

    # concatenate query embeddings with prompt embeddings
    inputs_embeds = self.get_input_embeddings()(input_ids)
    inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

    outputs = generate_lm(
        self=self.language_model,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask, # original func is until here
        max_length=64,
        min_length=16,
        num_beams=20,
        no_repeat_ngram_size=3,
        length_penalty=0.5,
        prune_factor=50,
        sat_tolerance=2,
        beta=0,
        early_stop=1.5,
        **generate_kwargs
    )

    return outputs