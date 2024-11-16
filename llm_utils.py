

import os
from dotenv import load_dotenv
import numpy as np
import pickle
import pandas as pd
import json
import re
from typing import Dict, List, Tuple

from openai import OpenAI
from anthropic import Anthropic 

# load_dotenv()
# oai = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
from settings import *
oai = OpenAI(api_key = OPENAI_API_KEY)

ant = Anthropic()
ant.api_key = os.getenv('ANTHROPIC_API_KEY')

def gen_oai(messages, model='gpt-4o', temperature=1, max_attempts = 3):
    if model is None:
        model = 'gpt-4o'
    attempts = 0
    while attempts < max_attempts:
        try:
            response = oai.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                max_tokens=2000
            )
            content = response.choices[0].message.content
            token_usage = response.usage
            input_tokens = token_usage.prompt_tokens
            output_tokens = token_usage.completion_tokens
            
            # Current pricing
            input_cost_per_1k = 0.005  # Cost per 1k input tokens
            output_cost_per_1k = 0.015  # Cost per 1k output tokens
            
            # Calculate costs
            input_cost = (input_tokens / 1000) * input_cost_per_1k
            output_cost = (output_tokens / 1000) * output_cost_per_1k
            total_cost = input_cost + output_cost
            
            # Print detailed cost breakdown
            print(f"API call cost breakdown:")
            print(f" - Input tokens: {input_tokens} tokens ($ {input_cost:.4f})")
            print(f" - Output tokens: {output_tokens} tokens ($ {output_cost:.4f})")
            print(f" - Total cost: $ {total_cost:.4f}")

            # Check if content is empty or only whitespace
            if content.strip() == "":
                attempts += 1
                print(f"Attempt {attempts}: Received empty or whitespace response. Retrying...")
                continue
            
            return content
        except Exception as e:
            print(f"Error generating completion on attempt {attempts + 1}: {e}")
            attempts += 1
            if attempts >= max_attempts:
              continue
    return ""

def gen_o1(messages, temperature=1):
  try:
    response = oai.chat.completions.create(
      model="gpt-4-0125-preview",
      temperature=temperature,
      messages=messages,
      max_tokens=2000
    )
    content = response.choices[0].message.content
    token_usage = response.usage
    input_tokens = token_usage.prompt_tokens
    output_tokens = token_usage.completion_tokens
    
    # Current pricing for o1-preview
    input_cost_per_1k = 0.015  # Cost per 1k input tokens
    output_cost_per_1k = 0.06  # Cost per 1k output tokens
    
    # Calculate costs
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    total_cost = input_cost + output_cost
    
    # Print detailed cost breakdown
    print(f"API call cost breakdown:")
    print(f" - Input tokens: {input_tokens} tokens ($ {input_cost:.4f})")
    print(f" - Output tokens: {output_tokens} tokens ($ {output_cost:.4f})")
    print(f" - Total cost: $ {total_cost:.4f}")

    return content
  except Exception as e:
    print(f"Error generating completion: {e}")
    raise e

def simple_gen_oai(prompt, model='gpt-4o', temperature=1):
  messages = [{"role": "user", "content": prompt}]
  return gen_oai(messages, model)

def gen_ant(messages, model='claude-3-5-sonnet-20240620', temperature=1, 
            max_tokens=1000):
  if model == None:
    model = 'claude-3-5-sonnet-20240620'
  try:
    response = ant.messages.create(
      model=model,
      max_tokens=max_tokens,
      temperature=temperature,
      messages=messages
    )
    content = response.content[0].text
    return content
  except Exception as e:
    print(f"Error generating completion: {e}")
    raise e

def simple_gen_ant(prompt, model='claude-3-5-sonnet-20240620'):
  messages = [{"role": "user", "content": prompt}]
  return gen_ant(messages, model)

# Prompt utils

# Prompt inputs
def fill_prompt(prompt, placeholders):
  for placeholder, value in placeholders.items():
    placeholder_tag = f"!<{placeholder.upper()}>!"
    if placeholder_tag in prompt:
      prompt = prompt.replace(placeholder_tag, str(value))
  return prompt

def make_output_format(modules):
  output_format = "Output Format:\n{\n"
  for module in modules:
    if 'name' in module and module['name']:
      output_format += f'    "{module["name"].lower()}": "<your response>",\n'
  output_format = output_format.rstrip(',\n') + "\n}"
  return output_format

def modular_instructions(modules):
    '''
    given some modules in the form

    name (optional, makes it a step)
    instruction (required)

    make the whole prompt
    '''
    prompt = ""
    step_count = 0
    for module in modules:
      if 'name' in module:
        # print(module)
        step_count += 1
        prompt += f"Step {step_count} ({module['name']}): {module['instruction']}\n"
      else:
        prompt += f"{module['instruction']}\n"
    prompt += "\n"
    prompt += make_output_format(modules)
    return prompt

# Prompt outputs
def parse_json(response, target_keys=None):
  json_start = response.find('{')
  json_end = response.rfind('}') + 1
  cleaned_response = response[json_start:json_end].replace('\\"', '"')
  
  try:
    parsed = json.loads(cleaned_response)
    if target_keys:
      parsed = {key: parsed.get(key, "") for key in target_keys}
    return parsed
  except json.JSONDecodeError:
    print("Tried to parse json, but it failed. Switching to regex fallback.")
    print(f"Response: {cleaned_response}")
    parsed = {}
    for key_match in re.finditer(r'"(\w+)":\s*', cleaned_response):
      key = key_match.group(1)
      if target_keys and key not in target_keys:
        continue
      value_start = key_match.end()
      if cleaned_response[value_start] == '"':
        value_match = re.search(r'"(.*?)"(?:,|\s*})', 
                                cleaned_response[value_start:])
        if value_match:
          parsed[key] = value_match.group(1)
      elif cleaned_response[value_start] == '{':
        nested_json = re.search(r'(\{.*?\})(?:,|\s*})', 
                                cleaned_response[value_start:], re.DOTALL)
        if nested_json:
          try:
            parsed[key] = json.loads(nested_json.group(1))
          except json.JSONDecodeError:
            parsed[key] = {}
      else:
        value_match = re.search(r'([^,}]+)(?:,|\s*})', 
                                cleaned_response[value_start:])
        if value_match:
          parsed[key] = value_match.group(1).strip()
    
    if target_keys:
      parsed = {key: parsed.get(key, "") for key in target_keys}
    return parsed
  

# end-to-end generation and parsing
def mod_gen(modules: List[Dict], placeholders: Dict, target_keys = None) -> Dict:
  prompt = modular_instructions(modules)
  filled = fill_prompt(prompt, placeholders)
  # print(filled)
  response = simple_gen_oai(filled)
  if len(response) == 0:
    print("Error: response was empty")
    return {}
  if target_keys == None:
    target_keys = [module["name"].lower() for module in modules if "name" in module]
  parsed = parse_json(response, target_keys)
  return parsed
