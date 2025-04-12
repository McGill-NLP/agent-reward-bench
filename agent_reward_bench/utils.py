from math import e
import os

KNOWN_MODEL_COSTS = {
    'gpt-4o-mini':       {'input': 0.15, 'output': 0.6, 'cached': 0.075, 'denom': 1_000_000},
    'gpt-4o':            {'input': 2.50, 'output': 10., 'cached': 1.250, 'denom': 1_000_000},
    'claude-3.7-sonnet': {'input': 3.00, 'output': 15., 'cached': 0.300, 'denom': 1_000_000},
}

class CostEstimator:
    def __init__(self, input_price, output_price, denominator, cached_price=None):
        self.input_price = input_price
        self.output_price = output_price
        self.denominator = denominator
        self.cached_price = cached_price if cached_price is not None else input_price
        self.total_cost = 0
    
    def estimate_cost(self, completion, finegrained=False):
        if isinstance(completion, dict):
            usage = completion['usage']
        elif completion.usage is None:
            if finegrained:
                return {
                    'input_price': 0,
                    'output_price': 0,
                    'cached_price': 0,
                    'total_price': 0
                }
            else:
                return 0
        else:
            usage = completion.usage.model_dump()

        input_tokens = usage['prompt_tokens']
        output_tokens = usage['completion_tokens']
        cached_tokens = usage.get('cached_tokens', 0)
        output_tokens -= cached_tokens

        input_price = input_tokens * self.input_price / self.denominator    
        output_price = output_tokens * self.output_price / self.denominator 
        cached_price = cached_tokens * self.cached_price / self.denominator 
        total_price = (input_price + output_price + cached_tokens)

        if finegrained:
            return {'input_price': input_price, 'output_price': output_price, 'total_price': total_price, 'cached_price': cached_price}
        else:
            return total_price

    @classmethod
    def from_dict(cls, d):
        return cls(d['input_price'], d['output_price'], d['denominator'])


    @classmethod
    def from_model_name(cls, model_name, zero_if_unknown=True):
        if '/' in model_name:
            company, _, model_name = model_name.partition('/')
            
        for model, costs in KNOWN_MODEL_COSTS.items():
            if model in model_name:
                return cls(costs['input'], costs['output'], costs['denom'])
        
        if zero_if_unknown:
            return cls(0, 0, 1)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def increment_cost(self, completion):
        cost = self.estimate_cost(completion, finegrained=False)
        self.total_cost += cost
        return self.total_cost

    def reset(self):
        self.total_cost = 0
    

def select_env_vars_by_provider(provider):
    """
    Selects the environment variables that are specific to a provider
    """
    if provider == "together":
        api_key = 'TOGETHER_API_KEY'
        base_url = 'TOGETHER_BASE_URL'
    elif provider == "openai":
        api_key = 'OPENAI_API_KEY'
        base_url = 'OPENAI_BASE_URL'
    elif provider == 'vllm':
        api_key = 'VLLM_API_KEY'
        base_url = 'VLLM_BASE_URL'
    elif provider == 'openrouter':
        api_key = 'OPENROUTER_API_KEY'
        base_url = 'OPENROUTER_BASE_URL'
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return {'api_key': api_key, 'base_url': base_url}


def get_api_key_from_env_var(provider):
    """
    Gets the API key for a provider
    """
    return os.getenv(select_env_vars_by_provider(provider)['api_key'])

def get_base_url_from_env_var(provider):
    """
    Gets the base URL for a provider
    """
    return os.getenv(select_env_vars_by_provider(provider)['base_url'])