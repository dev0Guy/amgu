def queue_length(prev_meta_data, current_meta_data):
    return -sum(current_meta_data["get_lane_waiting_vehicle_count"].values())


def queue_length_squared(prev_meta_data, current_meta_data):
    return -sum(
        map(
            lambda x: x**2,
            current_meta_data["get_lane_waiting_vehicle_count"].values(),
        )
    )


def queue_length_pln(prev_meta_data, current_meta_data):
    if not current_meta_data["action_time"]:
        return 0
    time = current_meta_data["action_time"] - prev_meta_data["action_time"]
    return queue_length(prev_meta_data, current_meta_data) / time


def delta_queue(prev_meta_data, current_meta_data):
    cur_count = sum(current_meta_data["get_lane_waiting_vehicle_count"].values())
    prev_count = sum(prev_meta_data["get_lane_waiting_vehicle_count"].values())
    return prev_count - cur_count


def delta_queue_pln(prev_meta_data, current_meta_data):
    if not current_meta_data["action_time"]:
        return 0
    time = current_meta_data["action_time"] - prev_meta_data["action_time"]
    return delta_queue(prev_meta_data, current_meta_data) / time


def waiting_time(prev_meta_data, current_meta_data):
    return -sum(current_meta_data["vehicle_waiting_time"].values())


def delta_waiting_time(prev_meta_data, current_meta_data):
    return -sum(current_meta_data["vehicle_waiting_time"].values()) + sum(
        prev_meta_data["vehicle_waiting_time"].values()
    )
