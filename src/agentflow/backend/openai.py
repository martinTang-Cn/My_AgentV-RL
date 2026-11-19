from __future__ import annotations

from typing import List, Tuple, Dict, Any
from logging import Logger
import random
import time

from openai import OpenAI

from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed

from agentflow.core.interfaces import CanGenerate,SupportChatTemplate
from agentflow.utils.log_util import get_logger
from agentflow.utils.chat_template import is_chat_messages

class OpenaiBackend(CanGenerate):
    """Backend based on Openai API
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger: Logger = None,
        *,
        request_options: Dict[str, Any] | None = None,
        tool_observations_as_user: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        if logger:
            self.logger = logger
        else:
            self.logger = get_logger(config, __name__)
        self._parse_config()

        self.client = OpenAI(
            api_key=self.openai_config["api_key"],
            base_url=self.openai_config["url"],
        )
        self.max_concurrency = self.openai_config.get("max_concurrency", 4)
        self.request_options = dict(request_options) if request_options is not None else {
            "temperature": self.sampling_config.get("temperature", 1),
            "max_tokens": self.sampling_config.get("max_tokens", 1024),
        }
        self.tool_observations_as_user = tool_observations_as_user
        
    def _generate(self, prompts: List, extra: List[Dict] = None, *, on_completion=None, cancel_event=None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If raw str are given, they would be wrapped as user prompt for api requests.
            extra (List[Dict], optional): Extra info dicts. Defaults to None.
            on_completion: Optional callback receiving the number of choices in each completed response.
            cancel_event: Optional event that stops submission of further requests.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw":<full_api_response_object>}
        """
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        results = [""] * len(prompts)
        metas = [{}] * len(prompts)
        input_messages = prompts
        if all([isinstance(prompt,str) for prompt in prompts]):
            input_messages = [[{"role":"user","content":prompt}] for prompt in prompts]
        else:
            assert is_chat_messages(input_messages), "Prompt must be str or chat message"
        if len(input_messages) == 1:
            text, meta = self._request(input_messages[0])
            if on_completion is not None:
                on_completion(len(getattr(meta.get("raw"), "choices", ())))
            return [text], [meta]
        max_worker = min(len(prompts),self.max_concurrency)
        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            future_to_idx = {
                executor.submit(self._request, messages): idx
                for idx, messages in enumerate(input_messages)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx], metas[idx] = future.result()
                except Exception as e:
                    self.logger.error(e)
                    results[idx]=""
                if on_completion is not None:
                    on_completion(len(getattr(metas[idx].get("raw"), "choices", ())))
        return results, metas
    
    def generate(self, prompts: List, extra: List[Dict] = None, **kwargs) -> Tuple[List[str],List[Dict]]:
        """Generate sequences with gievn prompt list

        Args:
            prompts (List): Prompt list of chat messages or raw str. If raw str are given, they would be wrapped as user prompt for api requests.
            extra (List[Dict], optional): Extra info dicts. Defaults to None.

        Returns:
            Tuple[List[str],List[Dict]]: Generated sequences and any metainfo
                - The metainfo format: {"raw":<full_api_response_object>}
        """
        return self._generate(prompts, extra, **kwargs)
        
        
  
        
    def _request(self, messages):
        if self.tool_observations_as_user:
            # The agent uses XML tool calls, so there is no API tool_call_id to attach.
            messages = [
                {"role": "user", "content": f'Python tool observation:\n{message["content"]}'}
                if message.get("source") == "tool" or message["role"] == "tool"
                else {"role": message["role"], "content": message["content"]}
                for message in messages
            ]
        args = {
            **self.request_options,
            "model": self.backend_config["model_path"],
            "messages": messages,
        }
        
        max_retries = self.openai_config.get("max_retries",2)
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(**args)

                return resp.choices[0].message.content or "", {"raw":resp,"input_messages":messages}

            except Exception as e:
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500 and status not in {408, 409, 429}:
                    self.logger.warning("API rejected the request (%s): %s", status, e)
                    return "", {"error_status": status, "error_message": str(e)}

                msg = f"[API attempt {attempt}/{max_retries}] Error: {e}"
                if self.logger:
                    self.logger.warning(msg, exc_info=True)

                if attempt == max_retries:
                    final_msg = f"API Failed after repeating {max_retries} times, aborting."
                    if self.logger:
                        self.logger.error(final_msg)
                    return "", {}

                sleep_time = 1 * (2 ** (attempt - 1))

                jitter = sleep_time * 0.1
                time_to_sleep = sleep_time + (jitter * (2 * random.random() - 1))
                if self.logger:
                    self.logger.info(f"Sleeping {time_to_sleep:.2f}s before retry…")
                time.sleep(time_to_sleep)
                continue
        return "", {}
        
    
    def _parse_config(self):
        backend_config = self.config["backend"]
        sampling_config = backend_config["sampling"]
        openai_config = backend_config["openai"]
        self.backend_name = "openai"
        self.backend_config = backend_config
        self.sampling_config = sampling_config
        self.openai_config = openai_config
