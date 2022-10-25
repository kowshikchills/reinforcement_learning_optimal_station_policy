
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym_env.envs import ground_env as ge
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN, A2C, PPO
import plotly.graph_objects as go
from os.path import exists
import os
import glob
import random

path = 'models/station_policy'
with st.sidebar:
    selected_menu = option_menu(
        menu_title="Kowshik's RL Project",  
        options=["Train Agent", "Blog", "DQN Algorithm", 'Contact Me'],  
        icons=["house", "bezier2", "lightning", 'envelope'],  
        menu_icon="cast", 
        default_index=0
    )


def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False
    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    if selected_menu == "Train Agent":
        st.header('RL Optimal Station Policy')
        st.warning('Select Parameters To Train Agent',icon="🕹")
        first, second, third = st.columns(3)
        first.info('Rewards',icon="⬆️")
        number1 = first.number_input('Pick Target', 100, help= 'The reward agent gets When it picks the target')
        first.info('Penalty',icon="⬇️")
        number2 = first.number_input('Cosine Distance Scale', 100, help=  'Scale factor for: Farther is the agent to the target larger the penalty')
        number3 = first.number_input('Single Step', 0.01, help=  'Penalty for movement at each timestep')
        
        
        third.info('Time Params',icon="⏱")
        number4 = third.number_input('Time Between Subsequent Pickups', 100, help=  'Once the agent picks the target, the next target is delayed by this value')
        third.info('Done State',icon="✅")
        number5 = third.number_input('Maximum Allowed Time to Pick', 100, help=  'The agent ends the game if it fails to catch the target in 100 time steps')
        number6 = third.number_input('Total Game Time', 10000, help=  'Total length of game')
        rewards_dict = {'cosine_distance_scale': number1, 'penalty_for_one_step':number2,
        'penalty_for_one_step_when_ball_is_here': 0.1, 'reward_for_catch':number3,
        'time_between_catch_next_ball':int(number4),'game_end_total':int(number5),
        'game_end_pickup_time':int(number6)}
        second.image('models/img.png')
        rewards_dict = {'cosine_distance_scale': 100, 'penalty_for_one_step': 0.01,
        'penalty_for_one_step_when_ball_is_here': 0.1, 'reward_for_catch':100, 'time_between_catch_next_ball':100,
        'game_end_total':1000,'game_end_pickup_time':10000}

        reward_params = list(rewards_dict.values())
        
        first, second, third = st.columns([0.45,0.2,0.4])
        train_flag = second.button('Train Agent')
        if train_flag:
            with st.spinner('Hold on Champ! DQN is Running'):
                env = ge.playground_env(100,'center','inverse_radial',reward_params =  reward_params)
                model = DQN("MlpPolicy", env, verbose=0, tensorboard_log="sac",learning_rate=0.001)
                model.learn(total_timesteps=1e5, log_interval=4,)
                model.save(path)
                st.success('🎉 Agent is Trained and Saved !')

        if exists(path+'.zip'):
            st.warning('Select Parameters To Test Agent',icon="🕹")
            first, second, third = st.columns([0.45,0.2,0.4])
            length_of_test = first.number_input('Length of Test Game', 2000, help=  'Total length of game, keep it small')
            length_of_test = int(length_of_test)
            speed_of_gif = third.number_input('Speed of Output', 5, help=  'Control the speed of the video')
            first, second, third = st.columns([0.45,0.2,0.4])
            test_flag = second.button('Run Agent')
            if test_flag:
                with st.spinner('Agent is busy predicting'):
                    model =  DQN.load(path)
                    done = False
                    reward_params_test = reward_params
                    reward_params_test[-1] = length_of_test
                    env = ge.playground_env(100,'center','inverse_radial',reward_params = reward_params)
                    obs = env.reset()
                    action_profile = []
                    reward_profile = []
                    frames_ = []
                    X_station = []
                    Y_station = []
                    c = 0
                    while not done:
                        c = c + 1 
                        act = model.predict(obs)
                        obs_, reward, done, info = env.step(act[0])
                        obs = obs_
                        action_profile.append([act[0]])
                        reward_profile.append(reward)
                        if env.PG.generated:
                            frames_.append(go.Frame(data=[go.Scatter(x=[env.PG.pos[0], env.PG.ran_pos[0]], y=[env.PG.pos[1], env.PG.ran_pos[1]])]))
                        else:
                            frames_.append(go.Frame(data=[go.Scatter(x=[env.PG.pos[0]], y=[env.PG.pos[1]])]))
                            X_station.append(env.PG.pos[0])
                            Y_station.append(env.PG.pos[1])
                st.success('🎉 Prediction is done, now view your results !')
                fig = go.Figure(
                    data=[go.Scatter(mode='markers',marker=dict(size=[15, 40], color = ['red','green']), text=['🚶','⚽️'], marker_symbol= ['star-triangle-up','circle'])],
                    layout=go.Layout(
                        xaxis=dict(range=[0, 100], autorange=False),
                        yaxis=dict(range=[0, 100], autorange=False),
                        title="Reinforcement Learning Agent in Working",
                        updatemenus=[dict(buttons = [dict(
                                                        args = [None, {"frame": {"duration": speed_of_gif, 
                                                                                "redraw": False},
                                                                        "fromcurrent": True, 
                                                                        "transition": {"duration": 0}}],
                                                        label = "Play",
                                                        method = "animate") , {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }],
                                                type='buttons',
                                                showactive=False)]
                    ),
                    frames=frames_
                )
                fig.update_layout(width= 800, height= 800,xaxis=dict(showgrid=False),yaxis=dict(showgrid=False))
                st.plotly_chart(fig, use_container_width=False)
                col1, col2, col3 = st.columns(3)
                col1.metric('Total Reward', '', np.round(sum(reward_profile),2))
                from collections import Counter
                col2.metric('Total Pickups','' , len(np.array(reward_profile)[np.array(reward_profile)>0]))

    if selected_menu == 'DQN Algorithm':
        st.header('DQN Annotated Paper')
        import base64
        file_path  = 'models/2015-Playing-Atari-with-Deep-Reinforcement Learning.pdf'
        def show_pdf(file_path):
            with open(file_path,"rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
        show_pdf(file_path)

    if selected_menu == 'Blog':
        from pathlib import Path
        import base64
        def read_markdown_file(markdown_file):
            return Path(markdown_file).read_text()
        intro_markdown = read_markdown_file("1.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)

    if selected_menu == 'Contact Me':
        st.image('models/contactme.png')
        st.header(' ')
        st.write('I am kowshik Chilamkurthy. I graduated from IIT Madras. I develop AI & RL Products in a German Startup')
        st.success('You will know everything about me here 😎')
        with open("models/kowshik_germany.pdf", "rb") as pdf_file:
            PDFbyte = pdf_file.read()
        st.download_button(label="Download Resume",
                            data=PDFbyte,
                            file_name="kowshik_iitm.pdf",
                            mime='application/octet-stream')
        
        st.subheader('How to Contact me ?')
        st.write('Mail: kowshikchilamkurty@gmail.com')
        link='[🔗 Follow me on Linkedin](https://www.linkedin.com/in/kowshik-chilamkurthy-67a501113/)'
        st.markdown(link,unsafe_allow_html=True)

        pass