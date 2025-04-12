import logging

logger = logging.getLogger(__name__)
# set to info
logger.setLevel(logging.INFO)

class AssistantbenchImprovedBackend:
    def prepare(self):
        logger.info("Preparing AssistantBench backend.")
        from . import register
        logger.info("AssistantBench backend is ready.")