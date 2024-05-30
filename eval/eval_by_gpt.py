import os
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import eval_light, eval_topdial


class OpenAIChat(object):
    
    def __init__(self, 
                 prompt_template,
                 api_key_file,
                 api_base_url = "https://api.openai.com/v1/",
                 model: str = "gpt-4-turbo",
                 temperature: float = 0.7, 
                 max_tokens: int = 64
                 ):
        # load prompt template
        self.template = open(f'{prompt_template}', 'r').read()
        if "light" in prompt_template:
            self.template_name = "light"
        elif "topdial" in prompt_template:
            self.template_name = "topdial"
        else:
            logging.warning("Unknown prompt template")
        # load api key
        api_key = self._load_api(api_key_file)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=api_base_url)
    
    def _load_api(self, path: str, is_pool: bool = False):
        api_keys = []
        with open(path, 'r') as f:
            for line in f:
                key = line.strip()
                api_keys.append(key)
        if is_pool:
            return api_keys
        else:
            return api_keys[0]

    def _get_response(self, messages):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        response = completion.choices[0].message.content
        response = response.strip()
        return response

    def query(self, data_example):
        if self.template_name == "light":
            setting, character, conv_history = data_example["context"]
            response = data_example["response"]
            prompt = self.template.format(
                agent_character=character,
                setting=setting,
                dialogue_context=conv_history,
                response=response
            )
        elif self.template_name == "topdial":
            user_profile, target, conv_history = data_example["context"]
            target_act = target.split("<")[1].split(",")[0]
            target_topic=  target.split("<")[1].split(",")[1].split(">")[0]
            if "movie" in target_act.lower():
                agent_role = "a movie enthusiast who enjoys a variety of films"
            elif "music" in target_act.lower():
                agent_role = "a music enthusiast who enjoys a variety of music"
            elif "food" in target_act.lower():
                agent_role = "a foodie who enjoys delicious food"
            elif "poi" in target_act.lower():
                agent_role = "a food enthusiast who is interested in exploring different restaurants"
            else:
                raise ValueError("Invalid target action: {}".format(target_act))
            
            response = data_example["response"]
            prompt = self.template.format(
                agent_role=agent_role,
                target_act=target_act,
                target_topic=target_topic,
                user_profile=user_profile,
                dialogue_context=conv_history,
                response=response
            )
        else:
            raise NotImplementedError

        messages = []
        messages.append({"role": "user", "content": prompt})
        score = None
        while True:
            response = self._get_response(messages)
            try:
                score = float(response)
                break
            except ValueError:
                print("Invalid score: {}".format(response))
                continue
        assert score is not None
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, default=None)
    parser.add_argument("--prompt_template", type=str, default="prompt/eval_light.txt")
    parser.add_argument("--max_history", type=int, default=4, help="Max number of history utterances to use, 0 denotes all to use.")
    parser.add_argument('--model', type=str, default='gpt-4-turbo')
    parser.add_argument('--api_key_file', type=str, default='openai_api_key.txt')
    parser.add_argument('--api_base_url', type=str, default="https://api.openai.com/v1/")
    args = parser.parse_args()
    
    openai_chat = OpenAIChat(args.prompt_template, args.api_key_file,
                             api_base_url=args.api_base_url, model=args.model)
    if openai_chat.template_name == "light":
        data_examples = eval_light.parse_data(args.gold_file, args.eval_file, max_history=args.max_history)
    elif openai_chat.template_name == "topdial":
        data_examples = eval_topdial.parse_data(args.gold_file, args.eval_file, max_history=args.max_history)
    else:
        raise NotImplementedError

    eval_scores = []
    save_name = args.eval_file.split('/')[-1].split('.')[0].replace('_output', '') + '_gpt_scores.jsonl'
    save_path = os.path.join(os.path.dirname(args.eval_file), save_name)
    with open(save_path, 'w', encoding='utf-8') as fw:
        for example in tqdm(data_examples):
            score = openai_chat.query(example)
            eval_scores.append(score)
            obj = {
                'dialog_id': example['dialog_id'],
                'round_id': example['round_id'],
                'consist_score': score,
            }
            fw.write(json.dumps(obj) + '\n')
            fw.flush()
    print("Saved consist. score to %s" % save_path)
    
    avg_prob = np.average(eval_scores) 
    print("Average consist. score: %f" % avg_prob)