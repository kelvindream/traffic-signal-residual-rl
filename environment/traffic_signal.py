import numpy as np
from gym import spaces


class TrafficSignal:
    def __init__(
        self,
        ts_id: str,
        yellow_time: int,
        simulation_time: float,
        delta_rs_update_time: int,
        sumo
    ):
        self.ts_id = ts_id
        self.yellow_time = yellow_time
        self.simulation_time = simulation_time
        self.delta_rs_update_time = delta_rs_update_time
        # reward_state_update_time
        self.rs_update_time = 0
        self.sumo = sumo
        self.green_phase = None
        self.yellow_phase = None
        self.decision_started = False
        self.end_time = 0
        self.all_phases = self.sumo.trafficlight.getAllProgramLogics(ts_id)[0].phases
        self.all_green_phases = [phase for phase in self.all_phases if 'y' not in phase.state]
        self.lanes_id = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.ts_id)))
        self.lanes_length = {lane_id: self.sumo.lane.getLength(lane_id) for lane_id in self.lanes_id}
        # 状态归一化参数
        self.state_min = np.zeros(4 * len(self.lanes_id), dtype=np.float32)
        self.state_max = np.ones(4 * len(self.lanes_id), dtype=np.float32)
        self.state_max[:len(self.lanes_id)] = 1.0  # 密度最大值
        self.state_max[len(self.lanes_id):2*len(self.lanes_id)] = 15.0  # 速度最大值（m/s）
        self.state_max[2*len(self.lanes_id):3*len(self.lanes_id)] = 100.0  # 排队车辆数最大值
        self.state_max[3*len(self.lanes_id):] = 100.0  # 平均等待时间最大值（s）
        self.observation_space = spaces.Box(
            low=np.zeros(4 * len(self.lanes_id), dtype=np.float32),
            high=np.ones(4 * len(self.lanes_id), dtype=np.float32))
        self.action_space = spaces.Discrete(len(self.all_green_phases))
        # last_measure for calculate reward
        self.last_measure = 0
        self.continue_reward = False
        self.dict_lane_veh = None
        # 历史交通数据，用于稀疏奖励计算
        self.prev_avg_speed = 0
        self.prev_avg_waiting_time = 0

    def change_phase(self, new_green_phase):
        """
        :param new_green_phase:
        :return: do_action -> the real action operated; if is None, means the new_green_phase is not appropriate,
        need to choose another green_phase and operate again
        """
        # yellow_phase has not finished yet

        self.decision_started = False

        # do_action mapping (int -> Phase)
        new_green_phase = self.all_green_phases[new_green_phase]
        current_time = self.sumo.simulation.getTime()
        if self.yellow_phase is not None:
            if current_time >= self.end_time:
                self.yellow_phase = None
                self.update_end_time()
                self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.green_phase.state)
                do_action = self.green_phase
            else:
                do_action = self.yellow_phase
        else:
            # if old_green_phase has finished
            if current_time >= self.end_time:
                if new_green_phase.state == self.green_phase.state:
                    self.rs_update_time = current_time
                    self.update_end_time()
                    self.decision_started = True
                    do_action = self.green_phase
                else:
                    # need to set a new plan(yellow + new_green)
                    yellow_state = ''
                    for s in range(len(new_green_phase.state)):
                        if self.green_phase.state[s] == 'G' and new_green_phase.state[s] == 'r':
                            yellow_state += 'y'
                        else:
                            yellow_state += self.green_phase.state[s]
                    self.yellow_phase = self.sumo.trafficlight.Phase(self.yellow_time, yellow_state)
                    self.sumo.trafficlight.setRedYellowGreenState(self.ts_id, self.yellow_phase.state)
                    self.green_phase = new_green_phase
                    # P4修复（2026-08-31）：原值 current+yellow+delta 导致reward在t+12s
                    # 才返回且携带的是t+2s的旧评分（start已在黄灯结束时消费），
                    # 经验(s,a,s',r)中r反映的是上一相位表现——credit assignment错位。
                    # 改为黄灯结束即返回：reward=切换时刻新相位指向排队最大方向的
                    # 即时评价（与MaxPressure决策语义对齐）。
                    self.rs_update_time = current_time + self.yellow_time
                    self.update_end_time()
                    self.decision_started = True
                    do_action = self.yellow_phase
            else:
                do_action = self.green_phase

        if do_action is None:
            return None

        # do_action mapping (Phase -> int)
        if 'y' in do_action.state:
            do_action = -1
        else:
            for i, green_phase in enumerate(self.all_green_phases):
                if do_action.state == green_phase.state:
                    do_action = i
                    break

        return do_action

    def update_end_time(self):
        current_time = self.sumo.simulation.getTime()
        if self.yellow_phase is None:
            self.end_time = current_time + self.green_phase.duration
        else:
            self.end_time = current_time + self.yellow_time

    def compute_reward(self, start, do_action):
        update_reward = False
        current_time = self.sumo.simulation.getTime()
        if current_time >= self.rs_update_time:
            # set rs_update_time unreachable
            self.rs_update_time = self.simulation_time + self.delta_rs_update_time
            update_reward = True

        return self._reward_with_speed(start, update_reward, do_action)

    def _reward_with_speed(self, start, update_reward, do_action):
        """优化的奖励函数，结合等待车辆数、平均速度和平均等待时间"""
        if start:
            # 计算各方向等待车辆数
            self.dict_lane_veh = {}
            for lane_id in self.lanes_id:
                self.dict_lane_veh[lane_id] = self.sumo.lane.getLastStepHaltingNumber(lane_id)
            # merge wait_num by actions
            dict_action_wait_num = [self.dict_lane_veh['n_t_0'] + self.dict_lane_veh['s_t_0'],
                                    self.dict_lane_veh['n_t_1'] + self.dict_lane_veh['s_t_1'],
                                    self.dict_lane_veh['e_t_0'] + self.dict_lane_veh['w_t_0'],
                                    self.dict_lane_veh['e_t_1'] + self.dict_lane_veh['w_t_1']]
            best_action = np.argmax(dict_action_wait_num)
            
            # 1. 基础奖励：基于等待车辆数
            if best_action == do_action:
                # 选择最优动作的奖励，根据等待车辆数的多少调整奖励大小
                max_wait_num = max(dict_action_wait_num)
                if max_wait_num > 50:
                    base_reward = 1.0  # 严重拥堵时选择最优动作奖励更大
                elif max_wait_num > 20:
                    base_reward = 0.5  # 中度拥堵
                else:
                    base_reward = 0.2  # 轻度拥堵
            else:
                # 选择非最优动作的惩罚，同样根据等待车辆数调整
                max_wait_num = max(dict_action_wait_num)
                if max_wait_num > 50:
                    base_reward = -1.0  # 严重拥堵时选择非最优动作惩罚更大
                elif max_wait_num > 20:
                    base_reward = -0.5  # 中度拥堵
                else:
                    base_reward = -0.2  # 轻度拥堵
            
            # 2. 速度奖励：平均速度的函数
            avg_speed = self.get_average_speed()
            # 将平均速度映射到[0, 1.0]范围内，速度越快奖励越高
            speed_reward = min(avg_speed / 10, 1.0)  # 假设最大速度约为10m/s
            
            # 3. 等待时间惩罚：平均等待时间的函数
            avg_waiting_time = self.get_average_waiting_time()
            # 将等待时间映射到[-1.0, 0]范围内，等待时间越长惩罚越大
            waiting_penalty = max(-avg_waiting_time / 100, -1.0)  # 100秒对应-1.0惩罚
            
            # 4. 排队长度惩罚：平均排队长度的函数
            avg_queue_length = self.get_average_queue_length()
            # 将排队长度映射到[-1.0, 0]范围内，排队越长惩罚越大
            queue_penalty = max(-avg_queue_length / 500, -1.0)  # 500米对应-1.0惩罚
            
            # 5. 稀疏奖励：当交通状况显著改善时给予额外奖励
            sparse_reward = 0
            # 检查是否有显著的交通改善
            if hasattr(self, 'prev_avg_speed') and hasattr(self, 'prev_avg_waiting_time'):
                speed_improvement = avg_speed - self.prev_avg_speed
                waiting_improvement = self.prev_avg_waiting_time - avg_waiting_time
                
                if speed_improvement > 1.0 and waiting_improvement > 10:
                    sparse_reward = 2.0  # 交通状况显著改善
                elif speed_improvement < -1.0 and waiting_improvement < -10:
                    sparse_reward = -2.0  # 交通状况显著恶化
            
            # 更新历史数据
            self.prev_avg_speed = avg_speed
            self.prev_avg_waiting_time = avg_waiting_time
            
            # 综合奖励：平衡各项奖励和惩罚
            self.last_measure = (base_reward + 
                               speed_reward * 0.3 + 
                               waiting_penalty * 0.2 + 
                               queue_penalty * 0.2 + 
                               sparse_reward * 0.3)

        if update_reward:
            return self.last_measure
        else:
            return None

    def compute_next_state(self):
        current_time = self.sumo.simulation.getTime()
        if current_time >= self.rs_update_time:
            state = self.compute_state()
            return state
        else:
            return None

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane_id) / (self.lanes_length[lane_id] / vehicle_size_min_gap))
                for lane_id in self.lanes_id]
    
    def get_average_speed(self):
        """获取所有车道的平均速度"""
        speeds = []
        for lane_id in self.lanes_id:
            avg_speed = self.sumo.lane.getLastStepMeanSpeed(lane_id)
            if avg_speed > 0:  # 仅考虑有车辆的车道
                speeds.append(avg_speed)
        return sum(speeds) / len(speeds) if speeds else 0
    
    def get_average_queue_length(self):
        """获取所有车道的平均排队长度"""
        queue_lengths = []
        for lane_id in self.lanes_id:
            queue_length = self.sumo.lane.getLastStepHaltingNumber(lane_id) * 7.5  # 每辆车占用7.5米
            queue_lengths.append(queue_length)
        return sum(queue_lengths) / len(queue_lengths)
    
    def get_average_waiting_time(self):
        """获取所有车辆的平均等待时间"""
        total_waiting_time = 0
        total_vehicles = 0
        
        # 限制最大处理车辆数，避免计算时间过长
        max_vehicles = 500
        processed_vehicles = 0
        
        for lane_id in self.lanes_id:
            if processed_vehicles >= max_vehicles:
                break
                
            lane_vehicles = self.sumo.lane.getLastStepVehicleIDs(lane_id)
            lane_vehicle_count = len(lane_vehicles)
            
            # 计算该车道可以处理的车辆数
            remaining_vehicles = max_vehicles - processed_vehicles
            process_count = min(lane_vehicle_count, remaining_vehicles)
            
            # 只处理部分车辆，提高计算效率
            for veh_id in lane_vehicles[:process_count]:
                total_waiting_time += self.sumo.vehicle.getWaitingTime(veh_id)
            
            total_vehicles += process_count
            processed_vehicles += process_count
        
        return total_waiting_time / total_vehicles if total_vehicles else 0
    
    def compute_state(self):
        """计算状态，包含更多特征并进行归一化"""
        # 1. 车道密度
        density = self.get_lanes_density()
        
        # 2. 车道平均速度
        speeds = []
        for lane_id in self.lanes_id:
            avg_speed = self.sumo.lane.getLastStepMeanSpeed(lane_id)
            speeds.append(avg_speed)
        
        # 3. 车道排队车辆数
        queue_lengths = []
        for lane_id in self.lanes_id:
            queue_length = self.sumo.lane.getLastStepHaltingNumber(lane_id)
            queue_lengths.append(queue_length)
        
        # 4. 车道平均等待时间
        waiting_times = []
        for lane_id in self.lanes_id:
            # 获取车道上的车辆
            lane_vehicles = self.sumo.lane.getLastStepVehicleIDs(lane_id)
            if lane_vehicles:
                # 计算该车道上车辆的平均等待时间
                lane_waiting_time = 0
                for veh_id in lane_vehicles[:10]:  # 限制每车道最多处理10辆车
                    lane_waiting_time += self.sumo.vehicle.getWaitingTime(veh_id)
                avg_waiting_time = lane_waiting_time / min(len(lane_vehicles), 10)
            else:
                avg_waiting_time = 0
            waiting_times.append(avg_waiting_time)
        
        # 组合所有特征
        state = np.concatenate([
            np.array(density),
            np.array(speeds),
            np.array(queue_lengths),
            np.array(waiting_times)
        ], dtype=np.float32)
        
        # 状态归一化
        state = (state - self.state_min) / (self.state_max - self.state_min + 1e-8)
        state = np.clip(state, 0.0, 1.0)
        
        return state
