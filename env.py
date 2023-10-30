import numpy as np
import gym
import wandb
from PySimpleGUI import Window, Button, Graph, TEXT_LOCATION_TOP_LEFT
from process import Process,RRQueue,FCFSQueue
STATE_SPACE_SIZE = 10

class Gui:
    COLORS = ["purple", "lightblue", "red", "green", "blue", "brown", "grey", "pink", "black", "yellow"]
    X1 = 100
    X2_OFF = 7
    Y1 = 10
    Y2_OFF = 15

    def __init__(self, size_x, size_y):
        self.graph = Graph(canvas_size=(size_x, size_y), graph_bottom_left=(0, size_y), graph_top_right=(size_x, 0),
                           key='graph')
        self.layout = [[self.graph], [Button('Exit')]]
        self.window = Window('MultiLevelFeedbackQueue', self.layout, finalize=True)

    def draw_process_rect(self, queue_num, queue_ticks, process_id):
        color = self.COLORS[process_id % len(self.COLORS)]
        y1_d = self.Y1 + 25 * queue_num
        x1_d = self.X1 + queue_ticks * self.X2_OFF
        self.graph.draw_rectangle((x1_d, y1_d), (x1_d + self.X2_OFF, y1_d + self.Y2_OFF), fill_color=color,
                                  line_color=color)

    def draw_queue_header(self, queue_id, quantum=None):
        queue_type = 'FCFS' if quantum is None else f'RR : {quantum}'
        return self.graph.draw_text(f'Queue {queue_id} {queue_type}',
                                    (5, 10 + 25 * queue_id), text_location=TEXT_LOCATION_TOP_LEFT)

    def print_process_statistics(self, i, job, number_of_queues):
        y1 = 10 + 25 * (number_of_queues + i)
        color = self.COLORS[job.process_id % len(self.COLORS)]
        self.graph.draw_rectangle((5, y1 + 3), (5 + 10, y1 + 13), fill_color=color, line_color=color)
        self.graph.draw_text(f"job arrival {job.arrival}, "
                             f"burst_time {job.burst_time}, "
                             f"turnaround_time: {job.statistics.turnaround}, "
                             f"wait: {job.statistics.wait}, "
                             f"response {job.statistics.response_time}",
                             (22, y1), text_location=TEXT_LOCATION_TOP_LEFT)

    def print_global_statistics(self, total_turnaround_time, total_wait, total_response,
                                total_jobs, number_of_queues, total_time, boost):
        boost = boost if boost > 0 else "no boost"
        self.graph.draw_text(f"Global Statistics\n"
                             f"average turnaround_time: {total_turnaround_time / total_jobs}\n"
                             f"average waiting_time: {total_wait / total_jobs}\n"
                             f"average response_time: {total_response / total_jobs}\n"
                             f"throughput: {total_jobs / total_time * 1000}ss\n"
                             f"boost jobs each: {boost}",
                             (5, 10 + 25 * (number_of_queues + total_jobs)),
                             text_location=TEXT_LOCATION_TOP_LEFT)
        self.graph.set_size((self.X1, 100 + 25 * (number_of_queues + total_jobs)))


def get_testcase(n, q, std_time=10, std_burst=5):
    """Generate a random test case of processes and time quantums."""
    testcase = [f'{int(abs(np.random.randn() * std_time)) + 1}:{int(abs(np.random.randn() * std_burst))}' for _ in range(n)]
    quantums = [8 * i for i in range(1, q + 1)]
    return testcase, quantums

def parse_jobs(jobs):
    return [Process(int(burst), int(arrival)) for job in jobs for burst, arrival in (job.split(":"),)]

def to_state_space(jobs):
    return np.convolve(jobs, np.ones(STATE_SPACE_SIZE + len(jobs) - 1) / STATE_SPACE_SIZE, 'valid')


class SchedulingEnv(gym.Env):
    def __init__(self, boost, number_of_queues, rendered=False):
        if rendered:
            self.gui = Gui(1000, 500)
        else:
            self.gui = None

        self.number_of_queues = number_of_queues
        self.boost = boost
        self.observation_space = gym.spaces.Box(np.array([0] * STATE_SPACE_SIZE * 2),
                                                np.array([np.inf] * STATE_SPACE_SIZE * 2),
                                                shape=(STATE_SPACE_SIZE * 2,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=1, high=100, shape=(self.number_of_queues - 1,), dtype=np.float32)

        self.queues = []
        self.job_list = []
        self.current_time = 0

    def init_queues(self, quantum_list):
        for i in range(self.number_of_queues - 1):
            self.queues.append(RRQueue(i, quantum_list[i]))
            if self.gui is not None:
                self.gui.draw_rr_queue_header(i, quantum_list[i])

        self.queues.append(FCFSQueue(self.number_of_queues - 1))
        if self.gui is not None:
            self.gui.draw_fcfs_queue_header(self.number_of_queues - 1)

        for i in range(self.number_of_queues - 1):
            self.queues[i].set_next_queue(self.queues[i + 1])

    def add_arrival_to_first_queue(self, process, priority):
        if process.arrival == self.current_time:
            self.queues[priority].add_process(process)

    def is_boost_available(self):
        return self.boost > 0 and self.current_time > 0

    def boost_jobs(self):
        if self.current_time % self.boost == 0:
            for queue in self.queues:
                queue.empty()
            for job in self.job_list:
                if not job.is_finished():
                    self.queues[0].add_process(job)

    def get_highest_non_empty_queue(self):
        for queue in self.queues:
            if not queue.is_empty():
                return queue

    def reset(self):
        jobs, self.quantum_list = get_testcase(10, self.number_of_queues)
        self.job_list = parse_jobs(jobs)
        self.queues = []
        self.init_queues(self.quantum_list)
        self.current_time = 0
        observation = np.append(to_state_space([0]), to_state_space([0]))
        print("Initial Quantums:", self.quantum_list)
        print("Jobs:", jobs)
        return observation

    def step(self, action):
        self.quantum_list = action
        for i in range(self.number_of_queues - 1):
            self.queues[i].quantum = self.quantum_list[self.queues[i].priority]

        pending_jobs = [job for job in self.job_list if not job.is_finished()]
        if not pending_jobs:
            return self.quantum_list, 0, True, {}

        for process in pending_jobs:
            self.add_arrival_to_first_queue(process, priority=0)

        if self.is_boost_available():
            self.boost_jobs()

        highest_queue = self.get_highest_non_empty_queue()
        reward = 0
        if highest_queue:
            process_id, reward = highest_queue.run_process(self.current_time)
            if self.gui is not None:
                self.gui.draw_process_rect(highest_queue.queue_id, self.current_time, process_id)

        total_time = [job.burst_time for job in self.job_list if job.arrival <= self.current_time]
        remaining_time = [job.time_left for job in self.job_list if job.arrival <= self.current_time]

        self.current_time += 1
        if not total_time:
            total_time = [0]
            remaining_time = [0]

        observation = np.append(to_state_space(remaining_time), to_state_space(total_time))
        return observation, reward, False, {}


  def print_stats(self):
    total_turnaround_time = []
    total_wait = []
    total_response = []
    total_processes = len(self.job_list)  # Total number of processes

    for job in self.job_list:
        total_response.append(job.statistics.response_time)
        total_turnaround_time.append(job.statistics.turnaround)
        total_wait.append(job.statistics.wait)

    total_time = self.current_time
    throughput = total_processes / total_time * 1000
    print("Total Time:", self.current_time)
    print("Average Turnaround Time:", np.mean(total_turnaround_time))
    print("Average Wait Time:", np.mean(total_wait))
    print("Average Response Time:", np.mean(total_response))
    print("Throughput:", throughput)

    print("Per Process Turnaround Time:", total_turnaround_time)
    print("Per Process Wait Time:", total_wait)
    print("Per Process Response Time:", total_response)


def log_stats(self):
    total_turnaround_time = []
    total_wait = []
    total_response = []
    total_processes = len(self.job_list)

    for job in self.job_list:
        total_response.append(job.statistics.response_time)
        total_turnaround_time.append(job.statistics.turnaround)
        total_wait.append(job.statistics.wait)
    total_time = self.current_time
    throughput = total_processes / total_time * 1000

    wandb.log({
        "Average_Turnaround_Time": np.mean(total_turnaround_time),
        "Average_Wait_Time": np.mean(total_wait),
        "Throughput": throughput,
        "Average_Response_Time": np.mean(total_response),
    })


  
        


    def log_stats(self):
        total_turnaround_time = []
        total_wait = []
        total_response = []
        total_processes = len(self.job_list)  

        for i, job in enumerate(self.job_list):
            total_response += [job.statistics.response_time]
            total_turnaround_time += [job.statistics.turnaround]
            total_wait += [job.statistics.wait]
        total_time=self.current_time
        throughput = total_processes / total_time*1000
    
        wandb.log({"Average_Turnaround_Time": np.mean(total_turnaround_time),
                   "Average_Wait_Time": np.mean(total_wait),
                   "Throughput:": throughput,  

                   "Average_Response_Time": np.mean(total_response),
                   })
    
    def render(self, mode='human'):
        if self.gui is None:
            print("This environment is not renderable. Initialize with rendered=True")
            return
        
        total_turnaround_time = total_wait = total_response = 0
        for i, job in enumerate(self.job_list):
            self.gui.print_process_statistics(i, job, self.number_of_queues)
            total_response += job.statistics.response_time
            total_turnaround_time += job.statistics.turnaround
            total_wait += job.statistics.wait
        
        total_jobs = len(self.job_list)
        self.gui.print_global_statistics(total_turnaround_time, total_wait, total_response,
                                         total_jobs, self.number_of_queues, self.current_time,self.boost)
        
        while True:
            event, values = self.gui.window.read()
            if event in (None, 'Exit'): 
                break
        self.gui.window.close()
    
    def close(self):
        pass
