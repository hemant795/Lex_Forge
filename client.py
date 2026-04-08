from openenv.core import EnvClient, SyncEnvClient
from models import LexAction, LexObservation

class LexForgeEnvClient(EnvClient):
    def _build_action(self, **kwargs) -> LexAction:
        return LexAction(**kwargs)
    def _parse_observation(self, data: dict) -> LexObservation:
        return LexObservation(**data)

class SyncLexForgeEnvClient(SyncEnvClient):
    def _build_action(self, **kwargs) -> LexAction:
        return LexAction(**kwargs)
    def _parse_observation(self, data: dict) -> LexObservation:
        return LexObservation(**data)
