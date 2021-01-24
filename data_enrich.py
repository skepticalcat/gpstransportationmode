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
            lat_1 = trajectory_frame["lat"][i-1]
            lat_2 = trajectory_frame["lat"][i]
            if lat_1 > 90:
                print("Faulty", lat_1)
                lat_1 /= 10
            if lat_2 > 90:
                print("Faulty", lat_2)
                lat_2 /= 10

            point_a = (lat_1, trajectory_frame["lon"][i-1])
            point_b = (lat_2, trajectory_frame["lon"][i])
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

    def set_sample_rate(self, trajectory_frame, min_sec_distance_between_points):
        i = 1
        indices_to_del = []
        deleted = 1
        while i < len(trajectory_frame)-deleted:
            ts1 = trajectory_frame["datetime"][i]
            ts2 = trajectory_frame["datetime"][i+deleted]
            delta = ts2-ts1
            if delta.seconds < min_sec_distance_between_points:
                deleted+=1
                indices_to_del.append(i)
                continue
            i+=deleted
            deleted = 1
        if indices_to_del:
            trajectory_frame.drop(trajectory_frame.index[indices_to_del],inplace=True)
            trajectory_frame.reset_index(inplace=True)

    def set_time_between_points(self, trajectory_frame):
        trajectory_frame["timedelta"] = 0
        for i, elem in trajectory_frame.iterrows():
            if i == 0:
                continue
            trajectory_frame["timedelta"][i] = (trajectory_frame["datetime"][i]-trajectory_frame["datetime"][i-1]).total_seconds()

    def get_enriched_data(self, from_pickle):
        if from_pickle:
            return pickle.load(open("data/raw_enriched.pkl", "rb"))

        traj = self.consolidate_trajectories()
        for elem in traj:
            self.set_sample_rate(elem, 5)
            self.set_time_between_points(elem)
            self.calc_dist_for_frame(elem)
            self.calc_speed_for_frame(elem)
            self.calc_accel_for_frame(elem)
        print("dumping")
        pickle.dump(traj, open("data/raw_enriched.pkl", "wb"))
        return traj


