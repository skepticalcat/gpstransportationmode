import pickle
import numpy as np
from geopy import distance

class DataEnrich:

    def __init__(self):
        pass

    def _load_raw_pickle(self):
        return pickle.load(open("data/raw_labeled.pkl","rb"))

    def consolidate_trajectories(self):
        raw_dfs = self._load_raw_pickle()
        trajectories = []
        for traj_of_person in raw_dfs:
            dfs_with_label = []
            for traj in traj_of_person:
                if "label" in traj.columns:
                    traj = traj.replace(to_replace='None', value=np.nan).dropna()
                    traj.reset_index(inplace=True)
                    dfs_with_label.append(traj)
            if dfs_with_label:
                trajectories.extend(dfs_with_label)
        return trajectories

    def _calc_speed(self, distance, ts_a, ts_b):
        time_delta = ts_b - ts_a
        if time_delta.total_seconds() == 0:
            return 0
        return distance / time_delta.total_seconds()  # m/s

    def _calc_accel(self, speed_a, speed_b, ts_a, ts_b):
        time_delta = ts_b - ts_a
        speed_delta = speed_b - speed_a
        if time_delta.total_seconds() == 0:
            return 0
        return speed_delta / time_delta.total_seconds()  # m/s^2

    def calc_dist_for_frame(self, trajectory_frame):
        trajectory_frame["dist"] = 0
        for i, elem in trajectory_frame.iterrows():
            if i == 0:
                continue
            point_a = (trajectory_frame["lat"][i-1], trajectory_frame["lon"][i-1])
            point_b = (trajectory_frame["lat"][i], trajectory_frame["lon"][i])
            if point_a[0] == point_b[0] and point_a[1] == point_b[1]:
                trajectory_frame["dist"][i] = 0
            else:
                trajectory_frame["dist"][i] = distance.distance((point_a[0], point_a[1]), (point_b[0], point_b[1])).m

    def calc_speed_for_frame(self, trajectory_frame):
            trajectory_frame["speed"] = 0
            for i, elem in trajectory_frame.iterrows():
                if i == 0:
                    continue
                trajectory_frame["speed"][i] = self._calc_speed(trajectory_frame["dist"][i],
                                                                trajectory_frame["datetime"][i-1],
                                                                trajectory_frame["datetime"][i]
                                                                )

    def calc_accel_for_frame(self, trajectory_frame):
        trajectory_frame["accel"] = 0
        for i, elem in trajectory_frame.iterrows():
            if i == 0:
                continue
            trajectory_frame["accel"][i] = self._calc_accel(trajectory_frame["speed"][i-1],
                                                            trajectory_frame["speed"][i],
                                                            trajectory_frame["datetime"][i - 1],
                                                            trajectory_frame["datetime"][i]
                                                            )
    def get_enriched_data(self):
        traj = self.consolidate_trajectories()
        for elem in traj:
            self.calc_dist_for_frame(elem)
            self.calc_speed_for_frame(elem)
            self.calc_accel_for_frame(elem)
        return traj


