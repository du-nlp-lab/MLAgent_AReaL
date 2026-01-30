from openai.types.chat import ChatCompletion

from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.core import workflow_context
from areal.experimental.openai import ArealOpenAI
from areal.tools.dsbench_executor import DSBenchExecutor
from areal.utils import logging, stats_tracker

logger = logging.getLogger("DSBenchWorkflow")


class DSBenchAgent:
    def __init__(self, gconfig, reward_fn, executor, max_turns=3):
        self.gconfig = gconfig
        self.max_turns = max_turns
        self.executor = executor
        self.async_reward_fn = AsyncRewardWrapper(reward_fn)

    async def run_agent(self, data, client: ArealOpenAI):
        messages = data["messages"].copy()
        work_dir = data.get("work_dir")
        reward = 0.0

        for t in range(self.max_turns):
            # 1. 模型推理 (Assistant Turn)
            response: ChatCompletion = await client.chat.completions.create(
                messages=messages,
                **self.gconfig.to_openai_args_dict(),
            )
            message = response.choices[0].message
            messages.append(message)

            # 2. 执行代码并获取反馈
            code_text = message.content
            exec_res = self.executor.execute(code_text, work_dir)
            observation = exec_res["observation"]

            # 3. 计算本轮奖励
            reward = await self.async_reward_fn(code_text=code_text, **data)
            client.set_reward(response.id, reward)

            if reward > 0 or t == self.max_turns - 1:
                break
            else:
                # 4. 注入报错信息进行多轮修正
                messages.append(
                    {
                        "role": "user",
                        "content": f"Execution Feedback:\n{observation}\nPlease fix the error.",
                    }
                )
        return reward


class MultiturnDSWorkflow(RolloutWorkflow):
    def __init__(
        self, reward_fn, gconfig, tokenizer, export_style="concat", max_turns=3
    ):
        from areal.utils.dynamic_import import import_from_string
        from areal.utils.hf_utils import load_hf_tokenizer

        self.tokenizer = load_hf_tokenizer(tokenizer)
        if isinstance(reward_fn, str):
            reward_fn = import_from_string(reward_fn)

        self.export_style = export_style
        self.agent = DSBenchAgent(
            gconfig=gconfig.new(n_samples=1),
            reward_fn=reward_fn,
            executor=DSBenchExecutor(),
            max_turns=max_turns,
        )

    async def arun_episode(self, engine, data):
        client = ArealOpenAI(
            engine=engine, tokenizer=self.tokenizer, chat_template_type="concat"
        )
        reward = await self.agent.run_agent(data=data, client=client)
        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=reward)

        client.apply_reward_discount(turn_discount=0.9)  # 应用多轮折扣
        return client.export_interactions(style=self.export_style)
