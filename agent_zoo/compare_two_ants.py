import os, gym, roboschool
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
config = tf.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1,
    device_count = { "GPU": 0 } )
sess = tf.InteractiveSession(config=config)

from RoboschoolAnt_v1_2017jul   import ZooPolicyTensorflow as PolAnt

PRINT_WEIGHTS = True
INDIV_SIZE = 12488

stadium = roboschool.scene_stadium.MultiplayerStadiumScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)
stadium.zero_at_running_strip_start_line = True

participants = []

original_weightfile = 'RoboschoolAnt_v1_2017jul.weights'
original_weights = {}
evolved_weightfile = 'Elite_Individual_ExperimentA3.weights'
evolved_weights = {}

exec(open(original_weightfile).read(), original_weights)
exec(open(evolved_weightfile).read(), evolved_weights)

#add original roboschool ant
lane = 0
env_id, PolicyClass = ("RoboschoolAnt-v1", PolAnt)
env = gym.make(env_id)
env.unwrapped.scene = stadium   # if you set scene before first reset(), it will be used.
env.unwrapped.player_n = lane
name1 = "OriginalModel"
pi = PolicyClass("{}".format(name1), env.observation_space, env.action_space, original_weights)
participants.append( (env, pi) )

#add elite individual from experiment
lane += 2
env_id, PolicyClass = ("RoboschoolAnt-v1", PolAnt)
env = gym.make(env_id)
env.unwrapped.scene = stadium   # if you set scene before first reset(), it will be used.
env.unwrapped.player_n = lane
name2 = "EvolvedModelA3"
pi = PolicyClass("{}".format(name2), env.observation_space, env.action_space, evolved_weights)
participants.append( (env, pi) )

if PRINT_WEIGHTS:
    layerNames = ['weights_dense1_w', 'weights_dense1_b', 'weights_dense2_w', 'weights_dense2_b', 'weights_final_w',
                  'weights_final_b']
    shapes = {}
    for layer in layerNames:
        shapes[layer] = original_weights[layer].shape
    orig = []
    evol = []
    for layer in layerNames:
        if len(shapes[layer]) == 2:
            for l in  original_weights[layer]:
                for w in l:
                    orig.append(w)
            for l in evolved_weights[layer]:
                for w in l:
                    evol.append(w)
        else:
            for w in  original_weights[layer]:
                orig.append(w)
            for w in evolved_weights[layer]:
                evol.append(w)
    file_name = 'weightComparison_{}_vs_{}.csv'.format(name1, name2)
    with open(file_name, 'w') as writer:
        writer.write('{}, {}, {}\n'.format('weight_index', name1, name2))
        index = 0
        for i in range(INDIV_SIZE):
            writer.write('{}, {}, {}\n'.format(str(i), str(orig[i]), str(evol[i])))

episode_n = 0
video = False
while 1:
    stadium.episode_restart()
    episode_n += 1

    multi_state = [env.reset() for env, _ in participants]
    frame = 0
    restart_delay = 0
    if video: video_recorder = gym.monitoring.video_recorder.VideoRecorder(env=participants[0][0], base_path=("/tmp/demo_race_episode%i" % episode_n), enabled=True)
    while 1:
        still_open = stadium.test_window()
        multi_action = [pi.act(s, None) for s, (env, pi) in zip(multi_state, participants)]

        for a, (env, pi) in zip(multi_action, participants):
            env.unwrapped.apply_action(a)  # action sent in apply_action() must be the same that sent into step(),
        # some wrappers will not work

        stadium.global_step()

        state_reward_done_info = [env.step(a) for a, (env, pi) in zip(multi_action, participants)]
        multi_state = [x[0] for x in state_reward_done_info]
        multi_done  = [x[2] for x in state_reward_done_info]

        if video: video_recorder.capture_frame()

        if sum(multi_done)==len(multi_done):
            break

        frame += 1
        stadium.cpp_world.test_window_score("%04i" % frame)
        if not still_open: break
        #if frame==1000: break
        if frame == 200: break
    if video: video_recorder.close()
    if not still_open: break

