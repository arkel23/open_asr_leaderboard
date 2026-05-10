import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_random, retry_if_exception
from sarvamai import SarvamAI
from sarvamai.core.api_error import ApiError
import cohere
from dotenv import load_dotenv

load_dotenv()

'''
# ------------------ RATE LIMITER ------------------
class RateLimiter:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()


rate_limiter = RateLimiter(60)  # 60 RPM
'''


# ------------------ RETRY ------------------
def is_retryable_error(exception):
    return (
        isinstance(exception, ApiError)
        and exception.status_code in [429, 500]
    )


def get_prompt_msg(question: list, prediction: list):
    prompt = f""" Question: {question} Model Answer: {prediction} Evaluate the answer quality. Score from 1 to 3: 1 = completely wrong, 2 = partially correct, 3 = correct Return ONLY the number. """

    return [
        {"role": "system", "content": "you are an evaluation judge."},
        {"role": "user", "content": prompt},
    ]


@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_random(min=1.2, max=4),  
    stop=stop_after_attempt(5),
)
def sarvam_call(client, msg):
    '''
    rate_limiter.wait()  #  prevents burst

    return client.chat.completions(
        model="sarvam-30b",
        messages=msg,
        temperature=0.5,
        top_p=1
    )
    '''
    
    return client.chat.completions(
        model="sarvam-30b",
        messages=msg,
        temperature=0.5,
        top_p=1
    )
   

@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_random(min=1.2, max=4),  # smaller jitter
    stop=stop_after_attempt(5),
)
def cohere_call(co, msg):
    '''
    rate_limiter.wait()  #  prevents burst
    return co.chat(
        model="command-a-03-2025",
        messages=msg,
        temperature=0.5,
    )
    '''
    return co.chat(
        model="command-a-03-2025",
        messages=msg,
        temperature=0.5,
    )

def sarvam_llm_as_judge(client, question, prediction):
    msg = get_prompt_msg(question, prediction)
    #print("************* prompt msg************* : ", msg)

    try:
        response = sarvam_call(client, msg)
        #print("************ sarvam response *********: ", response.choices[0].message.content.strip())
        return response.choices[0].message.content.strip()

    except ApiError as e:
        print(f"Final failure: {e.status_code} - {e.body}")
        return None

def cohere_llm_as_judge(client, question, prediction):
    msg = get_prompt_msg(question, prediction)
    #print("************* prompt msg************* : ", msg)

    try:
        response = cohere_call(client, msg)
        #print("************ cohere response *********: ", response.message.content[0].text)
        return response.message.content[0].text

    except ApiError as e:
        print(f"Final failure: {e.status_code} - {e.body}")
        return None



def run_llm_as_judge(llm_model, questions, predictions):

    if llm_model=="sarvam":
        api_key = os.getenv("SARVAM_API_KEY")
        print("********** sarvam was chosen ********************")
        client = SarvamAI(api_subscription_key=api_key)
        fnct = sarvam_llm_as_judge
    else: 
        #cohere code
        api_key = os.getenv("COHERE_API_KEY")
        print(api_key)
        print("********** cohere was chosen ********************")
        client = cohere.ClientV2(api_key=api_key)
        fnct = cohere_llm_as_judge
        
    print("********** start of thread ********************")

    results = []
    '''
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(fnct, client, q, p)
            for q, p in zip(questions, predictions)
        ]

        for future in as_completed(futures):
            results.append(future.result())
    '''
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_input = {
            executor.submit(fnct, client, q, p): (q, p)
            for q, p in zip(questions, predictions)}

        for future in as_completed(future_to_input):
            q, p = future_to_input[future]
            result = future.result()

            results.append({
                "question": q,
                "prediction": p,
                "score": result
            })


    return results
