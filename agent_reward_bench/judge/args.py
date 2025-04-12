default_judge_args = {
    "temperature": 0.0,
    "max_completion_tokens": 1024,
    "seed": 0,
}

judge_args = {
    'functional': {
        "provider": "openai",
        "model_name": "functional",
        "use_screenshot": False,
        "use_axtree": False,
    },
    "aerc": {
        "provider": "openai",
        "model_name": "gpt-4o-2024-11-20",
        "use_screenshot": False,
        "use_axtree": False,
    },
    "aerv": {
        "provider": "openai",
        "model_name": "gpt-4o-2024-11-20",
        "use_screenshot": True,
        "use_axtree": False,
    },
    "nnetnav": {
        "provider": "vllm",
        "model_name": "meta-llama/Llama-3.3-70B-Instruct",
        "use_screenshot": False,
        "use_axtree": False,
    },
    'gpt-4o-screen': {
        'provider': 'openai',
        'model_name': 'gpt-4o-2024-11-20',
        'use_screenshot': True,
        'use_axtree': False,
    },
    'gpt-4o-axtree': {
        'provider': 'openai',
        'model_name': 'gpt-4o-2024-11-20',
        'use_screenshot': False,
        'use_axtree': True,
    },
    "gpt-4o-mini-both": {
        "provider": "openai",
        "model_name": "gpt-4o-mini-2024-07-18",
        "use_screenshot": True,
        "use_axtree": True,
    },
    'gpt-4o-mini-axtree': {
        'provider': 'openai',
        'model_name': 'gpt-4o-mini-2024-07-18',
        'use_screenshot': False,
        'use_axtree': True,
    },
    'gpt-4o-mini-screen': {
        'provider': 'openai',
        'model_name': 'gpt-4o-mini-2024-07-18',
        'use_screenshot': True,
        'use_axtree': False,
    },
    'gpt-4o-mini-none': {
        'provider': 'openai',
        'model_name': 'gpt-4o-mini-2024-07-18',
        'use_screenshot': False,
        'use_axtree': False,
    },
    'claude-3.7-sonnet-screen': {
        'provider': 'openrouter',
        'model_name': 'anthropic/claude-3.7-sonnet',
        'use_screenshot': True,
        'use_axtree': False,
    },
    'claude-3.7-sonnet-axtree': {
        'provider': 'openrouter',
        'model_name': 'anthropic/claude-3.7-sonnet',
        'use_screenshot': False,
        'use_axtree': True,
    },
    'qwen-2.5-vl-screen': {
        'provider': 'vllm',
        'model_name': 'Qwen/Qwen2.5-VL-72B-Instruct',
        'use_screenshot': True,
        'use_axtree': False,
    },
    'qwen-2.5-vl-axtree': {
        'provider': 'vllm',
        'model_name': 'Qwen/Qwen2.5-VL-72B-Instruct',
        'use_screenshot': False,
        'use_axtree': True,
    },
    'llama-3.3-70b-axtree': {
        'provider': 'vllm',
        'model_name': 'meta-llama/Llama-3.3-70B-Instruct',
        'use_screenshot': False,
        'use_axtree': True,
    },
}