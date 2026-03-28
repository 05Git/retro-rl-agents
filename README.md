# Project

Train RL agents to play retro fighting games, starting with SF2 and expanding to others.

# Notes

- Use uv for package management
- Use sphinx for docs

# TODO

- Add run summaries
- Add proper documentation
- Handle default settings for things like PPO more robustly
- Add method for adding env wrappers
    - Current plan: Env wrappers can be defined in their own field, and can be applied after loading the config model.