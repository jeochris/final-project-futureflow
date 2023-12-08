import torch
from torch.nn import functional as F
import numpy as np

from topK import topk_huggingface, ConstrainedHypothesis

class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams * 2
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs, num_met):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        #score = sum_logprobs / math.pow((5 + len(hyp) + 1) / 6.0, self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, num_met))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len=None):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            #cur_score = best_sum_logprobs / math.pow((5 + cur_len + 1) / 6.0, self.length_penalty)
            ret = self.worst_score >= cur_score
            return ret


def _generate_beam_search(
        self,
        input_ids,
        logits_processor,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        bos_token_id,
        pad_token_id,
        eos_token_id,
        decoder_start_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        constraints,
        prune_factor,
        sat_tolerance,
        beta,
        early_stop,
        model_specific_kwargs,
):
    """ Generate sequences for each example with beam search.
    """

    #logits_processor = LogitsProcessorList()

    # end condition
    cons_eos = constraints[0].eos()

    last_non_masked_idx = (torch.sum(attention_mask, dim=1) - 1).int()
    # start_idx = (last_non_masked_idx).view(-1, 1).repeat(1, self.config.vocab_size).unsqueeze(1).long()

    # init_length = cur_len
    # position_ids = torch.tensor([list(range(init_length)) for i in range(input_ids.shape[0])])
    # for i, position_ids_slice in enumerate(position_ids):
    #     position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]
    # position_ids = position_ids.to(input_ids.device)
    #print(position_ids)

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states\
    #past = (encoder_outputs, None) if encoder_outputs is not None else None

    # done sentences
    done = [False for _ in range(batch_size)]

    # init number of met clauses
    num_mets = [x.num_met() for x in constraints]

    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, **model_specific_kwargs
        )
        #model_inputs["attention_mask"] = attention_mask
        #model_inputs["position_ids"] = position_ids[:, -1].unsqueeze(-1) if past else position_ids

        # print(cur_len)
        #print(model_inputs)

        outputs = self(**model_inputs,
                       return_dict=True,
                       output_attentions=(self.generation_config.output_attentions),
                       output_hidden_states=(self.generation_config.output_hidden_states))  # (batch_size * num_beams, cur_len, vocab_size)
        next_token_logits = outputs.logits[:, -1, :]  # (batch_size * num_beams, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        # if _use_cache(self, outputs, use_cache):
        #     past = outputs[1]
        # TODO (PVP) still a bit hacky here - there might be a better solution
        next_token_logits = self.adjust_logits_during_generation(
            next_token_logits, cur_len=cur_len
        )
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        next_token_scores_processed = logits_processor(input_ids, next_token_scores)

        # scores = self.postprocess_next_token_scores(
        #     scores=scores,
        #     input_ids=input_ids,
        #     no_repeat_ngram_size=no_repeat_ngram_size,
        #     bad_words_ids=bad_words_ids,
        #     cur_len=cur_len,
        #     min_length=min_length,
        #     max_length=max_length,
        #     eos_token_id=eos_token_id,
        #     repetition_penalty=repetition_penalty,
        #     batch_size=batch_size,
        #     num_beams=num_beams,
        # )

        # assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
        #     scores.shape, (batch_size * num_beams, vocab_size)
        # )

        if do_sample:
            raise NotImplementedError
        else:
            next_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)  # (batch_size * num_beams, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            full_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            next_scores, next_tokens = torch.topk(full_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

            pick_scores, pick_tokens, constraints, num_mets = topk_huggingface(timestep=cur_len,
                                                                               batch_size=batch_size,
                                                                               beam_size=num_beams,
                                                                               vocab_size=vocab_size,
                                                                               pad_token_id=pad_token_id,
                                                                               prune_factor=prune_factor,
                                                                               sat_tolerance=sat_tolerance,
                                                                               beta=beta,
                                                                               inactive=np.zeros((batch_size, num_beams)),
                                                                               scores=full_scores,
                                                                               hypotheses=constraints,
                                                                               num_fill=2 * num_beams,
                                                                               early_stop=early_stop)

            next_scores = torch.tensor(pick_scores, dtype=next_scores.dtype, device=next_scores.device)
            next_tokens = torch.tensor(pick_tokens, dtype=next_tokens.dtype, device=next_tokens.device)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0, None, -1)] * num_beams)  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score, constraint, num_met) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], constraints[batch_idx], num_mets[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                sentence_end = token_id.item() in constraint.eos()
                # add to generated hypotheses if end of sentence or last iteration
                if ((eos_token_id is not None) and (token_id.item() == eos_token_id)) or sentence_end:
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        torch.cat((input_ids[effective_beam_id], token_id.view([1]))), beam_token_score.item(), num_met,
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id, constraint, num_met))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if were done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx][:beam_token_rank + 1].max().item(), cur_len=cur_len
            ) or not next_sent_beam

            if len(next_sent_beam) < num_beams:
                if next_sent_beam:
                    pad_candidate = next_sent_beam[-1]
                elif done[batch_idx]:
                    pad_candidate = (0, pad_token_id, 0, None, -1)
                else:
                    raise ValueError('impossible search state')
                next_sent_beam += [pad_candidate] * (num_beams - len(next_sent_beam))

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])
        constraints = [x[3] for x in next_batch_beam]
        num_mets = [x[4] for x in next_batch_beam]

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        
        model_specific_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_specific_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        cur_len = cur_len + 1

        # re-order internal states
        # if past is not None:
        #     past = _reorder_cache(past, beam_idx)

        # extend attention_mask for new generated input if only decoder
        # if self.config.is_encoder_decoder is False:
        #     attention_mask = torch.cat(
        #         [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        #     )

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).item() not in cons_eos for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            final_num_met = num_mets[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score, final_num_met)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
    output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences
    #output_num_return_sequences_per_batch = 3

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    # print(input_ids)
    # print(sent_lengths)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        #print(i)
        #hyps = sorted(hypotheses.beams, key=lambda x: x[0], reverse=True)[:5]
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: (x[2], x[0]), reverse=True)
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps[0][1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # print(sent_lengths)
    # print(best)

    # shorter batches are padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(next(self.parameters()).device)

    return decoded