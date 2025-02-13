EPISODES = 50
EPSILON_START = 1.0

def train(env, agent):

    epsilon = EPSILON_START
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        agent.update_target_network()
        print(f"Episode {episode}, Balance: {env.balance}, holdfings: {env.holdings}")
