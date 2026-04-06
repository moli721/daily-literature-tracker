"""Daily Literature Tracker Playground"""

import logging
from pathlib import Path

from evomaster.core import BasePlayground, register_playground


@register_playground("daily_literature_tracker")
class DailyLiteratureTrackerPlayground(BasePlayground):
    """Daily Literature Tracker Playground
    
    自动追踪特定方向的新文献，生成今日摘要简报。
    """

    def __init__(self, config_dir: Path = None, config_path: Path = None):
        if config_path is None and config_dir is None:
            config_dir = (
                Path(__file__).parent.parent.parent.parent
                / "configs" / "daily_literature_tracker"
            )
        super().__init__(config_dir=config_dir, config_path=config_path)
        self.logger = logging.getLogger(self.__class__.__name__)
