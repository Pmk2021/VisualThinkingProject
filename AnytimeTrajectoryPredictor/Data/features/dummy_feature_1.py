from AnytimeTrajectoryPredictor.Data import FeatureExtractor


class DummyFeatureExtractor(FeatureExtractor):
    def __init__(self, args):
        super().__init__(args)
        self.feature_type = "DUMMY_FEATURE"

    def compute_feature(self, frames, dt_per_frame):
        """
        A dummy feature extractor that returns a constant value for each frame.
        This is just for testing purposes and should be replaced with actual logic.
        """

        feature_values = {
            i: 1.0 for i in range(len(frames))
        }  # Replace with actual computation

        return feature_values
