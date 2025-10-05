from session.test import TestSession
from session.train import TrainSession
from session.base import SessionEnum


class SessionFactory:
    @staticmethod
    def create_session(config, **kwargs):
        if config["mode"] == SessionEnum.TRAIN:
            return TrainSession(config, **kwargs)
        elif config["mode"] == SessionEnum.TEST:
            return TestSession(config, **kwargs)
        else:
            # DemonstrationSession, EvaluationSession...
            raise ValueError(f"Unknown session type: {config}")
