import os, gym, roboschool
import numpy as np
import tensorflow as tf
import time
from agent_zoo.RoboschoolAnt_v1_2017jul  import ZooPolicyTensorflow as PolAnt


# This example shows inner workings of multiplayer scene, how you can run
# several robots in one process.
class Eval(object):

    def __init__(self):
        self.episode_n = 0
        self.writer = open('evalLog.csv', 'w')
        self.writer.write('block1, block2, block3, block4, block5, block6\n')

    def evaluate_individual(self, weights):
        config = tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1,
            device_count={"GPU": 0})
        sess = tf.InteractiveSession(config=config)

        gym.logger.set_level(40)
        possible_participants = [
            ("RoboschoolAnt-v1", PolAnt),
        ]

        # individual = ("RoboschoolAnt-v1", PolAnt),
        # stadium = roboschool.scene_stadium.MultiplayerStadiumScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)
        stadium = roboschool.scene_stadium.SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165 / 4, frame_skip=4)

        # Place Ant in the center of the stadium
        stadium.zero_at_running_strip_start_line = False

        fitness = None
        participants = []

        for lane in range(1):
            env_id, PolicyClass = possible_participants[0]
            env = gym.make(env_id)
            env.unwrapped.scene = stadium   # if you set scene before first reset(), it will be used.
            env.unwrapped.player_n = lane   # mutliplayer scenes will also use player_n
            pi = PolicyClass("mymodel%i" % lane, env.observation_space, env.action_space, weights)
            participants.append( (env, pi) )

        # episode_n = 0
        video = False
        inProgress = True


        while inProgress:

            stadium.episode_restart()

            self.episode_n += 1

            multi_state = [env.reset() for env, _ in participants]
            frame = 0
            restart_delay = 0
            #if video: video_recorder = gym.monitoring.video_recorder.VideoRecorder(env=participants[0][0], base_path=("/tmp/demo_race_episode%i" % self.episode_n), enabled=True)

            while 1:
                # still_open = stadium.test_window()

                start_block1 = time.process_time()
                multi_action = [pi.act(s, None) for s, (env, pi) in zip(multi_state, participants)]

                for a, (env, pi) in zip(multi_action, participants):
                    env.unwrapped.apply_action(a)  # action sent in apply_action() must be the same that sent into step(),
                # some wrappers will not work
                end_block1 = time.process_time()
                start_block2 = time.process_time()
                stadium.global_step()
                end_block2 = time.process_time()
                start_block3 = time.process_time()
                state_reward_done_info = [env.step(a) for a, (env, pi) in zip(multi_action, participants)]
                end_block3 = time.process_time()
                start_block4 = time.process_time()
                multi_state = [x[0] for x in state_reward_done_info]
                multi_done  = [x[2] for x in state_reward_done_info]
                end_block4 = time.process_time()
                #if video: video_recorder.capture_frame()

                start_block5 = time.process_time()
                if sum(multi_done)==len(multi_done):
                    break

                frame += 1
                stadium.cpp_world.test_window_score("%04i" % frame)
                end_block5 = time.process_time()
                # if not still_open: break
                start_block6 = time.process_time()
                if frame == 50:
                    inProgress = False
                    fitness = participants[0][0].unwrapped.body_xyz[0]
                    break
                end_block6 = time.process_time()
                b1 = end_block1 - start_block1
                b2 = end_block2 - start_block2
                b3 = end_block3 - start_block3
                b4 = end_block4 - start_block4
                b5 = end_block5 - start_block5
                b6 = end_block6 - start_block6
                self.writer.write('{}, {}, {}, {}, {}, {}\n'.format(b1, b2, b3, b4, b5, b6))

            #if video: video_recorder.close()
            # if not still_open: break

        sess.close()

        return fitness
