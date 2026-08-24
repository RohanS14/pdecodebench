from vllm import LLM, SamplingParams

def main():
    print('Loading QwQ-32B...', flush=True)
    llm = LLM(
        model='Qwen/QwQ-32B',
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=32768,
    )
    params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=10)
    messages = [[
        {'role': 'system', 'content': '/no_think'},
        {'role': 'user', 'content': (
            'You are analyzing a numerical simulation written in Python.\n\n'
            '<code>\nu = u + dt * d2u_dx2\n</code>\n\n'
            'What type of PDE is this simulation solving?\n\n'
            'A) wave\nB) heat\nC) navier-stokes\nD) burgers\n\n'
            'Output only the letter of the correct answer.'
        )},
    ]]
    print('Running inference...', flush=True)
    outputs = llm.chat(messages, sampling_params=params,
                       chat_template_kwargs={'enable_thinking': False})
    out = outputs[0]
    text = out.outputs[0].text
    lp_dict = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
    print(f'First token: {repr(text)}', flush=True)
    print(f'Top tokens: {[(v.decoded_token, round(v.logprob,3)) for v in list(lp_dict.values())[:6]]}', flush=True)
    letter_lps = {v.decoded_token.strip(): v.logprob for v in lp_dict.values() if v.decoded_token.strip() in {'A','B','C','D'}}
    print(f'Letter logprobs: {letter_lps}', flush=True)
    if letter_lps:
        print('SUCCESS: /no_think fix works')
    else:
        print('FAIL: thinking still leaking')
        import sys; sys.exit(1)

if __name__ == '__main__':
    main()
