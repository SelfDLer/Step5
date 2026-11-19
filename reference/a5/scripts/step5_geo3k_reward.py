"""Geo3K reward adapter for Step5's prefilled assistant thinking tag."""
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
from verl.utils.reward_score import geo3k


def score_geo3k(prompt_tail, response, ground_truth):
    """Include only the prefilled assistant tag in the format check."""
    prefilled = prompt_tail.rstrip().endswith("assistant\n<think>")
    format_text = "<think>" + response if prefilled else response
    acc = geo3k.acc_reward(response, ground_truth)
    fmt = geo3k.format_reward(format_text)
    return {"score": 0.9 * acc + 0.1 * fmt, "acc": acc, "format_pass": fmt}


class Step5Geo3KRewardManager(NaiveRewardManager):
    """Use prompt-aware formatting for Geo3K; delegate other data sources."""

    async def run_single(self, data):
        item = data[-1]
        if item.non_tensor_batch["data_source"] != "hiyouga/geometry3k":
            return await super().run_single(data)
        result = await self.loop.run_in_executor(None, self._score_item, item)
        return {"reward_score": result["score"], "reward_extra_info": result}

    def _score_item(self, item):
        meta, batch = item.non_tensor_batch, item.batch
        prompt, response = batch["prompts"], batch["responses"]
        if "prompt_len" in meta and "response_len" in meta:
            # AgentLoop sends individually right-padded prompts and responses.
            prompt = prompt[:int(meta["prompt_len"])]
            response = response[:int(meta["response_len"])]
        else:
            mask = batch["attention_mask"].bool()
            prompt = prompt[mask[:prompt.numel()]]
            response = response[mask[batch["prompts"].numel():]]
        tail = self.tokenizer.decode(prompt[-32:], skip_special_tokens=False)
        text = self.tokenizer.decode(response, skip_special_tokens=True)
        return score_geo3k(tail, text, meta["reward_model"]["ground_truth"])
