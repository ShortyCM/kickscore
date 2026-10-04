class ObservationBatch:
    @classmethod
    def create(cls, observations, fitters):
        raise NotImplementedError("observation batching is owned by the Rust model")
