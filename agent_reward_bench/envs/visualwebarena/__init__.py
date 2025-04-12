import base64
import logging
from io import BytesIO

from .task import GenericVisualWebArenaTask
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def image_data_uri_to_pillow(image_data_uri) -> Image.Image:
    # remove the 'data:image/png;base64,' part
    prefix, image_data = image_data_uri.split(',')
    im_format = prefix.split(';')[0].split('/')[1]
    image_data = base64.b64decode(image_data)
    im = Image.open(BytesIO(image_data))
    return im

def pil_image_to_data_uri(im: Image.Image):
    buffer = BytesIO()
    im.save(buffer, format='PNG')
    image_data = buffer.getvalue()
    image_data_uri = f'data:image/png;base64,' + base64.b64encode(image_data).decode()
    return image_data_uri


class ResizedVisualWebArenaTask(GenericVisualWebArenaTask):
    def setup(self, page):
        print("Running ResizedVisualWebArenaTask")
        goal, second = super().setup(page)

        # goal is a list of dict, where each dict has a key 'text' and 'type'
        # when 'type' is 'image_url', we want to resize that image url and return a smaller one
        for part in goal:
            # resize the image 
            if part['type'] == 'image_url':
                logger.info(f"Resizing image {part['image_url']['url'][:20]}")
                image_data_uri = part['image_url']['url']
                im = image_data_uri_to_pillow(image_data_uri)
                im.thumbnail((1280, 1280))
                part['image_url']['url'] = pil_image_to_data_uri(im)
    
        print("Completed ResizedVisualWebArenaTask")
        return goal, second

class ResizedVWABackend:
    def prepare(self):
        from browsergym.experiments.benchmark.utils import massage_tasks
        
        # # register environments
        # import browsergym.visualwebarena  # This is the old approach
        import agent_reward_bench.envs.visualwebarena.register

        # full reset the instance (requires environment variables properly set up)
        from .instance import VisualWebArenaInstance

        default_instance = VisualWebArenaInstance()
        # default_instance.full_reset()

        logging.info(
            f"Initiating VisualWebArena instance warm-up. Some tasks will be pre-loaded (massaged) to trigger some caching mechanisms and make the server more responsive."
        )
        massage_tasks(
            [
                f"visualwebarena.resized.{id}"
                for id in [
                    # 0,  # classifieds
                    # 33,  # classifieds
                    # 555,  # shopping
                    # 666,  # shopping
                    # 282,  # __REDDIT__/f/dataisbeautiful
                    # 305,  # __REDDIT__/f/memes/new
                    # 314,  # __REDDIT__/f/mildlyinteresting
                    # 317,  # __REDDIT__/f/Art/active
                    # 318,  # __REDDIT__/f/consoles
                    # 319,  # __REDDIT__/f/EarthPorn
                    # 410,  # __REDDIT__/f/food
                    # 411,  # __REDDIT__/f/food
                    # 427,  # __REDDIT__/f/EarthPorn
                    # 436,  # __REDDIT__/f/Art
                    # 440,  # __REDDIT__/f/EarthPorn
                ]
            ]
        )


def resized_task_metadata():
    from ...benchmarks import task_metadata

    task_df = task_metadata("visualwebarena")
    task_df['task_name'] = task_df['task_name'].str.replace('visualwebarena', 'visualwebarena.resized')
    task_df['depends_on'] = task_df['depends_on'].str.replace('visualwebarena', 'visualwebarena.resized')
    return task_df