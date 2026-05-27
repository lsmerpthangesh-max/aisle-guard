class BehaviorEngine:
    def __init__(self, fps=30):
        self.fps = fps
        self.person_states = {}

        self.shelf_dwell_frames = int(2.0 * fps)
        self.cart_check_frames = int(4.0 * fps)
        self.cart_overlap_min_frames = 3
        self.risk_threshold = 0.75

    def _init_person(self, track_id, frame_id):
        self.person_states[track_id] = {
            "state": "IDLE",
            "state_start": frame_id,
            "risk": 0.0,
            "cart_overlap_frames": 0
        }

    def update(
        self,
        track_id,
        in_shelf_zone,
        in_cart_zone,
        frame_id
    ):
        if track_id not in self.person_states:
            self._init_person(track_id, frame_id)

        s = self.person_states[track_id]
        state = s["state"]

        # IDLE
        if state == "IDLE":
            if in_shelf_zone:
                s["state"] = "INTERACTING_WITH_SHELF"
                s["state_start"] = frame_id

        # INTERACTING WITH SHELF
        elif state == "INTERACTING_WITH_SHELF":
            dwell = frame_id - s["state_start"]

            if not in_shelf_zone:
                s["state"] = "IDLE"
                s["risk"] *= 0.5

            elif dwell >= self.shelf_dwell_frames:
                s["state"] = "CART_CHECK"
                s["state_start"] = frame_id
                s["cart_overlap_frames"] = 0
                s["risk"] += 0.2

        # CART CHECK
        elif state == "CART_CHECK":
            if in_cart_zone:
                s["cart_overlap_frames"] += 1

                if s["cart_overlap_frames"] >= self.cart_overlap_min_frames:
                    s["state"] = "SAFE"
                    s["risk"] = max(s["risk"] - 0.4, 0.0)

            elif frame_id - s["state_start"] > self.cart_check_frames:
                s["state"] = "MOVING_AWAY"
                s["state_start"] = frame_id
                s["risk"] += 0.3

        # MOVING AWAY
        elif state == "MOVING_AWAY":
            s["risk"] += 0.25

            if s["risk"] >= self.risk_threshold:
                s["state"] = "RISK"

        # SAFE
        elif state == "SAFE":
            s["risk"] *= 0.9

        # RISK
        elif state == "RISK":
            pass

        return s["state"], s["risk"]
