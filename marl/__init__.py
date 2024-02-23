from gymnasium.envs.registration import register

register(
    id="ChineseCheckers-v0",
    entry_point="chinesecheckers:ChineseCheckersEnv",
)