import openai
from tenacity import retry, stop_after_attempt, wait_fixed


class GptPrompter(object):
    def __init__(self, gpt_model_name='gpt-3.5-turbo', add_additional_politeness_prompt=True,
                 max_generation_tokens=1024, temperature=1.0):
        self._gpt_model_name = gpt_model_name
        self._additional_politeness_prompt = ''
        if add_additional_politeness_prompt:
            self._additional_politeness_prompt = ' and in a way that does not escalate the tension in the conversation'
        self._max_generation_tokens = max_generation_tokens
        self._temperature = temperature

    def reset_param(self, param, value):
        self.__dict__.update({param: value})

    def generate_prompt_from_utts(self, utts):
        setting = 'Here is an ongoing conversation between two users, Alice and Bob:'
        convo = '\n'.join([f'{"Alice" if idx % 2 == 0 else "Bob"}: {utts[idx]}' for idx in range(len(utts))])
        prompt = f'Continue the above conversation meaningfully{self._additional_politeness_prompt}; ' \
                 f'what should {"Alice" if len(utts) % 2 == 0 else "Bob"} say?'
        return f'{setting}\n\n{convo}\n\n{prompt}\n{"Alice" if len(utts) % 2 == 0 else "Bob"}:'

    @retry(wait=wait_fixed(1), stop=stop_after_attempt(10))
    def complete(self, prompt):
        completion = openai.ChatCompletion.create(model=self._gpt_model_name, max_tokens=self._max_generation_tokens,
                                                  n=1, temperature=self._temperature, stop=None,
                                                  messages=[{'role': 'user', 'content': prompt}])
        return completion['choices'][0]['message']['content']

    def generate_prompt_from_utts_and_complete(self, utts):
        prompt = self.generate_prompt_from_utts(utts=utts)
        response = self.complete(prompt)
        return prompt, response
