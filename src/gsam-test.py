import os
import logging

from dds_cloudapi_sdk.tasks.gsam import TinyGSAMTask, BaseGSAMTask
from dds_cloudapi_sdk import Config
from dds_cloudapi_sdk import Client
from dds_cloudapi_sdk import TextPrompt

def test():
    """
    python -m dds_cloudapi_sdk.tasks.gsam
    """
    test_token = os.environ["DDS_CLOUDAPI_TEST_TOKEN"]

    logging.basicConfig(level=logging.INFO)

    config = Config(test_token)
    client = Client(config)
    task = TinyGSAMTask(
        "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/grounded_sam/iron_man.jpg",
        [TextPrompt(text="iron man")]
    )

    client.run_task(task)
    print(task.result)

    task = BaseGSAMTask(
        "https://algosplt.oss-cn-shenzhen.aliyuncs.com/test_files/tasks/grounded_sam/iron_man.jpg",
        [TextPrompt(text="iron man")]
    )

    client.run_task(task)
    print(task.result)


if __name__ == "__main__":
    test()
